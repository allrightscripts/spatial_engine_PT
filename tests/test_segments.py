from tests.support import config_path
import json
import tempfile
import unittest
from pathlib import Path
from engine.galerkin.run import SegmentedRun
from engine.archive import read_archive
ROOT=Path(__file__).resolve().parents[1]

def config(name='spectral_single.json'):
    return json.loads((config_path(name)).read_text())

class SegmentTests(unittest.TestCase):
    def test_one_segment_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(config(),folder).run()
            self.assertEqual(m['status'],'completed');self.assertEqual(len(m['segments']),1)
            a=read_archive(Path(folder)/m['segments'][0]['archive'])
            self.assertEqual(a['coefficients'].shape[1],24)
            self.assertEqual(float(a['time'][-1]),.2)

    def test_structural_birth_and_projection(self):
        import numpy as np
        from engine.galerkin.basis import TensorBasis
        from engine.archive import decode
        c=config();c['events']=[{'type':'nucleation','time':.1,'strength':.3,'geometry':{'type':'gaussian','center':[.5],'radius':.1},'parameters':{'id':'P_0','family':'P','field_parameters':{'growth':.2}}}]
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,folder).run();self.assertEqual(len(m['segments']),2)
            a,b=[read_archive(Path(folder)/seg['archive']) for seg in m['segments']]
            self.assertEqual(b['coefficients'].shape[1],2*a['coefficients'].shape[1])
            basis=TensorBasis.from_descriptor(decode(b,'basis_json'))
            old=basis.reconstruct(a['coefficients'][-1].reshape((1,)+basis.shape))
            new=basis.reconstruct(b['coefficients'][0].reshape((2,)+basis.shape))
            np.testing.assert_allclose(old[0],new[0],atol=1e-12)
            self.assertEqual(m['relations'][0]['to'],1)
            self.assertIn('l2_error',m['relations'][0]['projection'])

    def test_catastrophe_split(self):
        c=config('cultural_v01.json')
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,folder).run()
            self.assertEqual(m['status'],'completed')
            split=next(e for e in m['events'] if e['event_type']=='split')
            self.assertEqual(split['targets'],['T_0','T_1'])
            self.assertAlmostEqual(split['parameters']['mass_before'],split['parameters']['mass_after'],places=12)
            self.assertGreaterEqual(split['time'],.23+.15)
            self.assertEqual(m['segments'][0]['end_time'],.23)
            self.assertGreater(m['segments'][-1]['dof'],m['segments'][0]['dof'])
            for seg in m['segments']:read_archive(Path(folder)/seg['archive'])

    def test_duration_and_target_family(self):
        import numpy as np
        c=config();c['t_end']=.3;c['initial_fields'][0]['parameters']={'diffusion':0}
        c['events']=[{'type':'catastrophe','time':.05,'duration':.1,'strength':2.,'shape':'rectangle','center':[.5],'width':1.,'target_families':['T'],'effect_type':'rate','close_segment':True}]
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,folder).run()
            self.assertEqual(len(m['segments']),3)
            self.assertAlmostEqual(m['segments'][1]['end_time'],.15)
            self.assertEqual(m['events'][1]['parameters']['phase'],'end')
            a=read_archive(Path(folder)/m['segments'][-1]['archive'])
            self.assertAlmostEqual(a['coefficients'][-1,0],float(np.exp(-.2)),places=7)

    def test_reproducibility(self):
        import numpy as np
        c=config();c['t_end']=.3;c['nucleation']={'rate':15.,'amplitude':.2,'radius':.1,'parameters':{'growth':.3}}
        c['stochastic_catastrophe']={'rate':10.,'event':{'shape':'gaussian','radius':.1,'strength':.3,'targets':['*'],'effect_type':'impulse'}}
        with tempfile.TemporaryDirectory() as folder:
            first=Path(folder)/'a';second=Path(folder)/'b'
            a=SegmentedRun(c,first).run();b=SegmentedRun(c,second).run()
            self.assertEqual(a['events'],b['events']);self.assertEqual(a['genealogy'],b['genealogy'])
            self.assertTrue(any(e['event_type']=='nucleation' for e in a['events']))
            for x,y in zip(a['segments'],b['segments']):
                np.testing.assert_allclose(read_archive(first/x['archive'])['coefficients'],read_archive(second/y['archive'])['coefficients'],rtol=0,atol=0)

    def test_single_archive_and_run_viewer(self):
        from engine.visualization.archive import plot_archive
        from engine.visualization.spectral import plot_spectral
        c=config();c['boundary']={'type':'periodic'};c['solver']['modes']=[4];c['solver']['dealiasing']='2/3'
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,Path(folder)/'run').run()
            archive=Path(folder)/'run'/m['segments'][0]['archive']
            plot_archive(archive,frame_index=1,save=Path(folder)/'single.png')
            plot_spectral(Path(folder)/'run/manifest.json',frame_index=1,save=Path(folder)/'run.png')
            self.assertGreater((Path(folder)/'run.png').stat().st_size,1000)

    def test_cross_backend(self):
        import numpy as np
        from engine.simulation.solver import Simulation
        from engine.galerkin.provider import GalerkinProvider
        from engine.galerkin.basis import TensorBasis
        from engine.numerics.ode import integrate_provider
        c=config();c['events']=[];c['grid']['shape']=[80];c['boundary']={'type':'periodic'}
        c['initial_fields'][0]['parameters']={'diffusion':.01,'growth':.3,'capacity':2.}
        c['dt']=.001;c['t_end']=.1;c['save_dt']=.1
        finite=Simulation(c);x=finite.grid.coords[0];initial=1+.1*np.cos(2*np.pi*x);finite.u[0]=initial
        basis=TensorBasis(finite.grid,['fourier'],[8],'2/3')
        provider=GalerkinProvider(c,finite.registry,basis)
        spectral=integrate_provider(provider,basis.project(initial),0,.1,method='Radau')
        finite.run();field=basis.reconstruct(spectral['coefficients'][-1])
        np.testing.assert_allclose(finite.u[0],field,atol=3e-6)

    def test_sine_2d_and_original_cosine_viewer(self):
        import importlib.util
        c=config();c['grid']={'shape':[20,22],'lengths':[1.,1.]};c['dimension']=2
        c['boundary']={'type':'dirichlet','value':0};c['solver']['modes']=[3,3]
        c['t_end']=.05;c['solver']['max_step']=.01
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,Path(folder)/'sine').run()
            a=read_archive(Path(folder)/'sine'/m['segments'][0]['archive'])
            self.assertEqual(a['modes'].shape,(9,2))
            from engine.visualization.archive import plot_archive
            plot_archive(Path(folder)/'sine'/m['segments'][0]['archive'],frame_index=1,save=Path(folder)/'sine.png')
            c=config();c['t_end']=.05
            m=SegmentedRun(c,Path(folder)/'cosine').run()
            spec=importlib.util.spec_from_file_location('original_viewer',ROOT/'tests/fixtures/legacy/ARCHIVE_to_MATPLOT.py')
            viewer=importlib.util.module_from_spec(spec);spec.loader.exec_module(viewer)
            viewer.plot_archive(Path(folder)/'cosine'/m['segments'][0]['archive'],frame_index=1,save=Path(folder)/'original.png')

    def test_extinction_and_final_event_state(self):
        c=config();c['events']=[{'time':.1,'type':'extinction','targets':['T_root']}]
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,folder).run()
            self.assertEqual(m['termination_reason'],'no_active_fields')
            self.assertEqual(m['relations'][0]['to'],None)
            self.assertTrue((Path(folder)/'terminal_state.npz').is_file())
            from engine.visualization.spectral import ArchiveSequence
            sequence=ArchiveSequence(Path(folder)/'manifest.json')
            self.assertEqual(sequence.frame(len(sequence.frames)-1)[0],{})

    def test_analysis_orchestration(self):
        from engine.cli.analyze import analyze
        c=config();c['initial_fields'][0]['parameters']={'growth':1.,'capacity':1.,'diffusion':.01}
        with tempfile.TemporaryDirectory() as folder:
            m=SegmentedRun(c,folder).run()
            result=analyze(Path(folder)/m['segments'][0]['path'])
            self.assertLess(result['residual'],1e-8);self.assertEqual(result['stable'],24)
