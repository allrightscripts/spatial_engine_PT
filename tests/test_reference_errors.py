from tests.support import config_path
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from engine.simulation.grid import Grid
from engine.analysis.error_metrics import error_metrics
from engine.analysis.reference_solutions import reference_solution, register_reference, REFERENCES
from engine.analysis.reference_errors import analyze_reference
from engine.cli.diffusion import run_benchmark

ROOT=Path(__file__).resolve().parents[1]


class ErrorMetricTests(unittest.TestCase):
    def test_equality_and_offset_nonunit_domain(self):
        for shape,lengths in [([17],[3.5]),([8,13],[2.,3.])]:
            g=Grid(shape,lengths);ref=np.ones(g.shape)*2.5
            self.assertTrue(all(v==0 for v in error_metrics(ref,ref,g).values()))
            c=-.3;eps=1e-10;m=error_metrics(ref+c,ref,g,eps);V=np.prod(lengths)
            self.assertAlmostEqual(m['L2'],abs(c)*np.sqrt(V),places=13)
            self.assertAlmostEqual(m['rel_L2'],abs(c)*np.sqrt(V)/(2.5*np.sqrt(V)+eps),places=13)
            self.assertAlmostEqual(m['Linf'],abs(c),places=13)
            self.assertAlmostEqual(m['mass_abs'],abs(c)*V,places=13)
            self.assertAlmostEqual(m['mass_rel'],abs(c)*V/(2.5*V+eps),places=13)

    def test_zero_reference_and_invalid_epsilon(self):
        g=Grid([10],[4.]);z=np.zeros(g.shape);eps=1e-6
        m=error_metrics(np.ones(g.shape),z,g,eps)
        self.assertEqual(m['L2'],2.);self.assertEqual(m['rel_L2'],2./eps)
        self.assertEqual(m['mass_rel'],4./eps)
        self.assertTrue(all(v==0 for v in error_metrics(z,z,g).values()))
        for eps in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):error_metrics(z,z,g,eps)
        with self.assertRaises(ValueError):error_metrics(z[:2],z,g)
        with self.assertRaises(ValueError):error_metrics(z+np.nan,z,g)

    def test_reference_cosine_1d_2d_and_registry(self):
        for g,k in [(Grid([32],[2.]),[3]),(Grid([20,24],[2.,3.]),[2,1])]:
            p={'D':.07,'k':k,'amplitude':.6};t=.3
            eigen=np.pi**2*sum((a/L)**2 for a,L in zip(k,g.lengths))
            expected=.6*np.exp(-.07*eigen*t)*np.prod([np.cos(a*np.pi*x/L) for a,x,L in zip(k,g.mesh,g.lengths)],axis=0)
            np.testing.assert_allclose(reference_solution('diffusion_neumann_cosine',g,t,p),expected,atol=1e-15)
        @register_reference('constant_test')
        def constant(grid,t,params):return params['value']
        try:np.testing.assert_array_equal(reference_solution('constant_test',g,0,{'value':2}),2)
        finally:del REFERENCES['constant_test']
        for p in ({'D':-1},{'k':[-1,0]},{'k':[1.5,2]},{'k':[1]},{'D':float('nan')}):
            with self.assertRaises(ValueError):reference_solution('diffusion_neumann_cosine',g,0,p)

    def test_analytic_benchmark_all_frames_exports_and_cli(self):
        p={'D':.1,'k':[2]}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run_benchmark(root/'run',p,modes=[4],shape=[48],lengths=[2.])
            segment=root/'run/segment_000'
            report=analyze_reference(segment,'diffusion_neumann_cosine',p,output=root/'errors.csv')
            self.assertEqual(len(report['rows']),5)
            self.assertLess(max(r['L2'] for r in report['rows']),1e-9)
            with (root/'errors.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]),['time','field','L2','rel_L2','Linf','mass_abs','mass_rel'])
            self.assertEqual(len(rows),5)
            analyze_reference(segment,'diffusion_neumann_cosine',p,output=root/'errors.json')
            saved=json.loads((root/'errors.json').read_text());self.assertEqual(saved['basis']['modes'],[4])
            params=root/'ref.json';params.write_text(json.dumps(p))
            result=subprocess.run([sys.executable,'-m', 'engine', 'analyze','--segment',str(segment),
                '--reference','diffusion_neumann_cosine','--reference-params',str(params),'--field','u'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertTrue((segment/'errors.csv').exists())
            with self.assertRaises(ValueError):analyze_reference(segment,'diffusion_neumann_cosine',p,fields=['missing'])

    def test_smooth_diffusion_convergence(self):
        params=json.loads((config_path('reference_diffusion_smooth.json')).read_text())
        errors=[]
        with tempfile.TemporaryDirectory() as d:
            for modes in (4,8,16,32):
                root=Path(d)/str(modes)
                run_benchmark(root,params,modes=[modes],shape=[128],lengths=[2.],t_end=.2)
                report=analyze_reference(root/'segment_000','diffusion_neumann_cosine',params)
                errors.append(report['rows'][-1]['L2'])
        for previous,current in zip(errors,errors[1:]):self.assertLess(current,previous*.2)
        self.assertLess(errors[-1],1e-9)

    def test_fv_archive_reference_path(self):
        from engine.simulation.solver import Simulation
        from engine.archive import write_archive
        c={'seed':1,'grid':{'shape':[12],'lengths':[3.]},'dt':.01,'t_end':.02,'save_dt':.01,
           'initial_fields':[{'id':'u','family':'test','initial':{'value':2},'parameters':{'diffusion':.1}}]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fv.npz';write_archive(path,Simulation(c).run())
            report=analyze_reference(path,'diffusion_neumann_cosine',{'D':.1,'k':0,'amplitude':2})
            self.assertTrue(all(r['L2']==0 for r in report['rows']))
