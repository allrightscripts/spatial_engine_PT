from tests.support import config_path
"""3D numerical checks and end-to-end archive/render regression."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from engine.simulation.grid import Grid
from engine.galerkin.basis import TensorBasis
from engine.galerkin.provider import GalerkinProvider
from engine.galerkin.run import SegmentedRun
from engine.model.masks import evaluate_mask
from engine.events.splitting import components, SplitDetector
from engine.simulation.solver import Simulation
from engine.archive import read_archive, write_archive, decode

ROOT=Path(__file__).resolve().parents[1]

class ThreeDimensionalTests(unittest.TestCase):
    def test_tensor_roundtrip_and_laplacian(self):
        grid=Grid([12,14,16],[1.,1.3,1.7])
        for kinds in [('cosine',)*3,('sine',)*3,('fourier',)*3,('sine','cosine','fourier')]:
            with self.subTest(kinds=kinds):
                b=TensorBasis(grid,kinds,[3,3,3])
                a=np.random.default_rng(7).normal(size=(2,)+b.shape)
                np.testing.assert_allclose(b.project(b.reconstruct(a)),a,atol=2e-13)
                np.testing.assert_allclose(b.weak_divergence(b.gradient(a)),b.laplacian(a),atol=3e-11)
                other=TensorBasis.from_descriptor(b.descriptor())
                np.testing.assert_allclose(other.reconstruct(a),b.reconstruct(a))

    def test_fv_mass_and_z_derivative(self):
        g=Grid([8,10,12],[1.,1.5,2.])
        u=np.random.default_rng(9).random((1,)+g.shape)
        for kind in ['neumann','periodic']:
            result=g.transport(u,np.ones((1,1,1,1,1)),[{'type':kind}])
            self.assertAlmostEqual(float(result.sum()*g.volume),0.,places=11)
        z=g.mesh[2]
        g_z=g.gradient_faces(z,{'type':'neumann','value':0},2)
        np.testing.assert_allclose(g_z[:,:,1:-1],1.,atol=1e-14)

    def test_diffusion_solution_and_analysis(self):
        from engine.numerics.ode import integrate_provider
        from engine.numerics.rhs import evaluate_rhs
        import engine.numerics.newton as newton
        g=Grid([10,12,14],[1.,1.2,1.4]);b=TensorBasis(g,['cosine']*3,[2]*3)
        fields=[{'id':'u','family':'T','parameters':{'diffusion':.1},'bc':{'type':'neumann'}}]
        p=GalerkinProvider({'model':'cultural_v01'},fields,b)
        a=np.zeros(b.shape);a[0,0,0]=1.;a[1,1,1]=.2
        for method in ['DOP853','Radau']:
            data=integrate_provider(p,a.ravel(),0,.1,method=method,save_dt=.05)
            expected=a*np.exp(-.1*b.eigenvalues*.1)
            np.testing.assert_allclose(data['coefficients'][-1],expected.ravel(),atol=1e-8)
        order,params,compiled=newton.load_equations(p.source())
        J=newton.numerical_jacobian(lambda x:evaluate_rhs(compiled,0,x,params),a.ravel())
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(J)),np.sort(-.1*b.eigenvalues.ravel()),atol=1e-7)

    def test_masks_connectivity_and_split_conservation(self):
        g=Grid([8,10,12],[1.,1.,1.])
        for kind in ['sphere','ellipsoid','gaussian','box','cuboid','my_catastrophe']:
            spec={'type':kind,'center':[.5]*3,'radius':.3,'low':[.2]*3,'high':[.8]*3}
            mask=evaluate_mask(g,spec)
            self.assertEqual(mask.shape,g.shape);self.assertGreater(mask.sum(),0)
        u=np.zeros((4,5,6));u[0,0,0]=u[0,0,-1]=1
        self.assertEqual(len(components(u,.5,[])),2)
        self.assertEqual(len(components(u,.5,[2])),1)
        u[:]=0;u[1,1,1]=u[2,2,2]=1
        self.assertEqual(len(components(u,.5,[])),2)  # corners are not neighbours
        c=json.loads((config_path('basic_3d.json')).read_text())
        c['initial_fields'][0]['initial']={'value':1}
        sim=Simulation(c);initial=sim.u[0].copy()
        masks=[sim.grid.mesh[2]<.6,sim.grid.mesh[2]>.8]
        SplitDetector({}).split(sim,0,masks,.1)
        np.testing.assert_allclose(sim.u[1:].sum(axis=0),initial)

    def test_events_archive_fv_and_renderers(self):
        from engine.visualization.spectral import ArchiveSequence,plot_spectral
        from engine.visualization.viewer import plot_grid_archive
        from engine.visualization.archive import plot_archive
        c=json.loads((config_path('basic_3d.json')).read_text())
        c['grid']['shape']=[8,10,12];c['solver']['modes']=[3]*3
        c['t_end']=.12;c['save_dt']=.04
        c['events']=[
            {'type':'catastrophe','time':.03,'shape':'box','center':[.5,.6,.7],'width':[.2,1.2,1.4],'strength':.5,'targets':['T_root']},
            {'type':'landscape','time':.05,'geometry':{'type':'sphere','center':[.5,.6,.7],'radius':.2},'strength':.5,'duration':.02},
            {'type':'nucleation','time':.09,'strength':.1,'geometry':{'type':'gaussian','center':[.5,.6,.7],'radius':.2},'parameters':{'id':'P_0','family':'P','field_parameters':{'diffusion':.001}}}]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'run'/'manifest.json'
            m=SegmentedRun(c,manifest.parent).run()
            self.assertEqual(m['status'],'completed');self.assertGreaterEqual(len(m['segments']),4)
            seq=ArchiveSequence(manifest)
            vals,grid,_,_=seq.frame(-1)
            self.assertEqual(vals['P_0'].shape,(8,10,12))
            for seg in m['segments']:
                d=read_archive(manifest.parent/seg['archive']);self.assertEqual(d['modes'].shape[1],3)
            for view in (1,2,3):
                plot_spectral(manifest,view_dimension=view,frame_index=0,save=root/f's{view}.png',slice_fractions=(.2,.3,.7),grid_points=6)
                self.assertGreater((root/f's{view}.png').stat().st_size,1000)
            plot_spectral(manifest,save=root/'movie.gif',fps=5,color_scale='global',grid_points=4)
            archive=manifest.parent/m['segments'][0]['archive']
            plot_archive(archive,frame_index=0,save=root/'archive.png',grid_points=4)
            fv=copy.deepcopy(c);fv['events']=[];fv['initial_fields'][0]['initial']={'value':.4}
            fv_data=Simulation(fv).run();write_archive(root/'fv.npz',fv_data)
            for view in (1,2,3):
                plot_grid_archive(root/'fv.npz',view_dimension=view,frame_index=0,save=root/f'fv{view}.png',grid_points=4)

if __name__=='__main__':unittest.main()
