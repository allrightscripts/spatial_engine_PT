from tests.support import config_path
"""Integration checks: birth-profile and reference patches on the 3D base."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from engine import __version__
from engine.galerkin.run import SegmentedRun
from engine.galerkin.basis import TensorBasis
from engine.visualization.spectral import plot_spectral
from engine.analysis.reference_errors import analyze_reference
from engine.cli.diffusion import run_benchmark
from engine.archive import read_archive,decode

ROOT=Path(__file__).resolve().parents[1]

class Merge08Tests(unittest.TestCase):
    def test_3d_birth_projection_metadata_and_render(self):
        c=json.loads((config_path('sandbox_3d_inside_T.json')).read_text())
        c['grid']['shape']=[12,14,16];c['solver']['modes']=[4,4,4]
        c['t_end']=.1;c['save_dt']=.05;c['events']=[];c['split']['enabled']=False
        c['stochastic_events']['p_birth'].update(max_births=1,rate_parameters={'lambda0':10.,'alpha':0})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=SegmentedRun(c,root).run()
            self.assertEqual(m['status'],'completed');self.assertEqual(len(m['segments']),2)
            self.assertEqual(m['software_version'],__version__)
            a,b=[read_archive(root/s['archive']) for s in m['segments']]
            basis=TensorBasis.from_descriptor(decode(a,'basis_json'))
            T=np.maximum(basis.reconstruct(a['coefficients'][-1].reshape(basis.shape)),0)
            expected=basis.project((T/(T.max()+1e-12))**3).ravel()
            np.testing.assert_allclose(b['coefficients'][0,b['fields']=='P_0'],expected,atol=1e-12)
            self.assertEqual(m['events'][0]['parameters']['birth_profile'],'inside_T')
            plot_spectral(root/'manifest.json',frame_index=2,grid_points=5,save=root/'frame.png')
            self.assertGreater((root/'frame.png').stat().st_size,1000)

    def test_3d_exact_reference_export(self):
        p={'D':.1,'k':[1,1,1]}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            run_benchmark(root/'run',p,modes=[3,3,3],shape=[10,12,14],lengths=[1.,1.2,1.4])
            r=analyze_reference(root/'run/segment_000','diffusion_neumann_cosine',p,output=root/'errors.csv')
            self.assertEqual(r['grid_shape'],[10,12,14]);self.assertEqual(len(r['rows']),5)
            self.assertLess(max(x['L2'] for x in r['rows']),1e-9)
            self.assertTrue((root/'errors.csv').exists())
