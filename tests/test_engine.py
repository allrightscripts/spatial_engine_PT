from tests.support import config_path
"""Acceptance, numerical consistency, archive safety, and legacy regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.archive import read_archive,write_archive,validate,decode,reconstruct_grid,pack_events,iter_events
from engine.simulation.solver import Simulation
from engine.simulation.grid import Grid
from engine.events.splitting import components,SplitDetector
from engine.events.base import CatastropheEvent,ManualEvent,SabotageEvent,MutationEvent
from engine.events.operators import spatial_mask
from engine.model.definitions import Model

ROOT=Path(__file__).resolve().parents[1]
def config(name='catastrophe_1d.json'):
    return json.loads((config_path(name)).read_text())

class IntegrationTests(unittest.TestCase):
    def test_catastrophe_split_archive(self):
        sim=Simulation(config()); data=sim.run()
        self.assertEqual([f['id'] for f in sim.registry],['T_root','T_0','T_1'])
        split=next(e for e in sim.events if e['event_type']=='split')
        self.assertAlmostEqual(split['parameters']['mass_before'],split['parameters']['mass_after'],places=14)
        self.assertAlmostEqual(split['time'],.5)
        self.assertFalse(sim.registry[0]['active']); self.assertEqual(sim.registry[1]['generation'],1)
        self.assertEqual(sim.registry[1]['parameters'],sim.registry[0]['parameters'])
        self.assertGreater(data['time'][-1],split['time'])
        self.assertEqual(data['coefficients'].shape,(31,240))
        before=data['time']<split['time']; self.assertTrue(np.all(data['coefficients'][before,80:]==0))
        self.assertTrue(np.all(data['coefficients'][data['time']>=split['time'],:80]==0))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.npz'; write_archive(p,data); loaded=read_archive(p)
            np.testing.assert_array_equal(reconstruct_grid(loaded,-1),sim.u)
            with np.load(p,allow_pickle=False) as z:
                self.assertTrue(all(not z[k].dtype.hasobject for k in z.files))
        self.assertEqual(len(decode(data,'genealogy_json')),1)

    def test_reproducible_innovation_and_mutation(self):
        c=config('innovation_1d.json'); a=Simulation(c); b=Simulation(c)
        x=a.run(); y=b.run()
        for key in x: np.testing.assert_array_equal(x[key],y[key],err_msg=key)
        self.assertTrue(any(e['event_type']=='nucleation' for e in a.events))
        self.assertTrue(any(e['event_type']=='mutation' for e in a.events))
        p=a.registry[a.index('P_0')]
        self.assertFalse(p['active']); self.assertEqual(p['death_time'],1.4)
        self.assertEqual(p['family'],'P')
        self.assertIn('state_before_step',next(e['rng'] for e in a.events if e['event_type']=='nucleation'))

    def test_seed_changes_nucleation(self):
        c=config('innovation_1d.json');c['events']=c['events'][:1]
        a=Simulation(c);a.run();c['seed']+=1;b=Simulation(c);b.run()
        ga=[e['geometry'] for e in a.events if e['event_type']=='nucleation']
        gb=[e['geometry'] for e in b.events if e['event_type']=='nucleation']
        self.assertNotEqual(ga,gb)

    def test_2d_acceptance_and_viewer(self):
        from engine.visualization.archive import plot_archive
        sim=Simulation(config('catastrophe_2d.json'));data=sim.run()
        self.assertEqual(len(sim.registry),3)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.npz'; png=Path(d)/'view.png';write_archive(p,data)
            plot_archive(p,frame_index=15,save=png)
            self.assertGreater(png.stat().st_size,1000)

    def test_non_grid_aligned_schedule(self):
        c=config();c['events'][0]['time']=.303;c['events'][0]['parameters']['duration']=.117
        sim=Simulation(c);sim.run()
        cats=[e for e in sim.events if e['event_type']=='catastrophe']
        self.assertAlmostEqual(cats[0]['time'],.303);self.assertAlmostEqual(cats[1]['time'],.420)
        self.assertTrue(np.all(np.diff(sim.times)>0))

    def test_extinct_rows_cannot_receive_boundary_inflow(self):
        c=config();c['boundary']={'type':'dirichlet','value':1};c['events']=[{'type':'extinction','time':.1,'targets':['T_root']}]
        sim=Simulation(c);data=sim.run()
        self.assertTrue(np.all(data['coefficients'][data['time']>=.1]==0))

    def test_custom_reaction_and_functional(self):
        class ZeroReaction:
            def __call__(self,u,fields,grid,t): return np.zeros_like(u)
        c=config();c['events']=[]
        sim=Simulation(c,model=Model(reaction=ZeroReaction()));data=sim.run()
        np.testing.assert_allclose(data['observables_total_mass'],1)
        np.testing.assert_allclose(data['observables_functional'],.5)

    def test_innovation_growth_competition_and_extinction(self):
        from engine.model.definitions import LogisticCompetition
        c=config();sim=Simulation(c)
        sim.add_field('P_0','P',np.full(sim.grid.shape,.1),0,{'growth':1,'competition':.5})
        reaction=LogisticCompetition()
        interacting=reaction(sim.u,sim.registry,sim.grid,0)[1]
        alone=reaction(sim.u[1:],sim.registry[1:],sim.grid,0)[0]
        self.assertTrue(np.all(interacting>0));self.assertTrue(np.all(alone>interacting))
        c['events']=[];c['extinction_mass']=.5;c['initial_fields'][0]['initial']['value']=.001
        sim=Simulation(c);sim.run()
        self.assertFalse(sim.registry[0]['active']);self.assertEqual(sim.events[0]['event_type'],'extinction')

    def test_periodic_diffusion_solution(self):
        c=config();c['grid']['shape']=[32];c['boundary']={'type':'periodic'}
        c['initial_fields'][0]['parameters']={'diffusion':.01};c['events']=[]
        c['t_end']=.1;c['dt']=.001;c['save_dt']=.05
        sim=Simulation(c);x=sim.grid.mesh[0];sim.u[0]=1+.2*np.cos(2*np.pi*x)
        data=sim.run();exact=1+.2*np.cos(2*np.pi*x)*np.exp(-.01*4*np.pi**2*.1)
        np.testing.assert_allclose(sim.u[0],exact,atol=3e-5)
        np.testing.assert_allclose(data['observables_total_mass'],1,atol=1e-14)

    def test_manual_sabotage_mutation(self):
        sim=Simulation(config())
        sim.apply(ManualEvent(0,targets=['T_root'],strength=.2))
        sim.apply(SabotageEvent(0,source='P_agent',targets=['T_root'],strength=.5))
        np.testing.assert_allclose(sim.u,.6)
        sim.apply(MutationEvent(0,targets=['T_root'],parameters={'updates':{'growth':.7}}))
        self.assertEqual(sim.registry[0]['parameters']['growth'],.7)

class SpatialTests(unittest.TestCase):
    def test_linear_dirichlet_1d_2d(self):
        for shape in ([20],[15,12]):
            g=Grid(shape,[1]*len(shape));u=sum(g.mesh)
            # Test one-axis gradients with exact Dirichlet data for 1D;
            # 2D manufactured linear u=x, zero normal derivative on y faces.
            u=g.mesh[0]
            bc={'axes':[[{'type':'dirichlet','value':0},{'type':'dirichlet','value':1}]]+[[{'type':'neumann'},{'type':'neumann'}]]*(len(shape)-1)}
            rhs=g.transport(u[None],np.ones((1,1)+(1,)*len(shape)),[bc])
            np.testing.assert_allclose(rhs,0,atol=2e-12)

    def test_nonhomogeneous_neumann_signs(self):
        g=Grid([30],[1]);u=g.mesh[0]
        bc={'axes':[[{'type':'neumann','value':-1},{'type':'neumann','value':1}]]}
        np.testing.assert_allclose(g.transport(u[None],np.ones((1,1,1)),[bc]),0,atol=1e-12)
        bc={'type':'neumann','value':2}
        rhs=g.transport(np.zeros_like(u)[None],np.ones((1,1,1)),[bc])
        self.assertAlmostEqual(rhs.sum()*g.volume,4)

    def test_periodic_eigenmode_convergence(self):
        errors=[]
        for n in (20,40):
            g=Grid([n],[1]);u=np.sin(2*np.pi*g.mesh[0]);rhs=g.transport(u[None],np.ones((1,1,1)),[{'type':'periodic'}])[0]
            errors.append(np.linalg.norm(rhs+4*np.pi**2*u)/np.sqrt(n))
        self.assertGreater(errors[0]/errors[1],3.9)

    def test_mass_conservation_cross_mobility_2d(self):
        g=Grid([10,12],[1,2]);u=np.random.default_rng(19).random((2,10,12))
        L=np.array([[1,.2],[.2,2]])[:,:,None,None]*(1+u[0])[None,None]
        for bc in ({'type':'periodic'},{'type':'neumann','value':0}):
            rhs=g.transport(u,L,[bc,bc]);np.testing.assert_allclose(rhs.sum(axis=(1,2)),0,atol=1e-10)

    def test_different_bc_per_field(self):
        g=Grid([10],[1]);u=np.zeros((2,10));L=np.eye(2)[:,:,None]
        rhs=g.transport(u,L,[{'type':'dirichlet','value':1},{'type':'neumann','value':0}])
        self.assertGreater(rhs[0].sum(),0);self.assertEqual(rhs[1].sum(),0)

    def test_periodic_components(self):
        u=np.zeros((8,7));u[0,2:5]=1;u[-1,2:5]=1
        self.assertEqual(len(components(u,.5,[])),2)
        self.assertEqual(len(components(u,.5,[0,1])),1)

    def test_reconnection_resets_isolation(self):
        c=config();sim=Simulation(c);d=sim.detector
        sim.u[0,35:45]=0;d.check(sim,0);d.check(sim,.1)
        sim.u[0]=1;d.check(sim,.15)
        sim.u[0,35:45]=0;d.check(sim,.2);d.check(sim,.3)
        self.assertEqual(len(sim.registry),1)
        d.check(sim,.4);self.assertEqual(len(sim.registry),3)

    def test_masks(self):
        g=Grid([20,20],[1,1])
        for kind in ('circle','ellipse','rectangle','gaussian'):
            geom={'type':kind,'center':[.5,.5],'radii':[.2,.1],'low':[.3,.3],'high':[.7,.7]}
            m=spatial_mask(g,geom);self.assertEqual(m.shape,g.shape);self.assertGreater(m.sum(),0);self.assertLessEqual(m.max(),1)

class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=Simulation(config()).run()
    def test_reject_objects_nonfinite_shapes(self):
        for key,value in [('extra',np.array([{}],object)),('coefficients',np.full_like(self.data['coefficients'],np.nan)),('fields',np.array(['wrong']))]:
            d=dict(self.data);d[key]=value
            with self.assertRaises(ValueError):validate(d)
    def test_reject_birth_corruption(self):
        d=copy.deepcopy(self.data);d['coefficients'][1,-1]=.1
        with self.assertRaises(ValueError):validate(d)
    def test_packed_event_roundtrip(self):
        events=list(iter_events(self.data));packed=pack_events(events,json_limit=0)
        d={k:v for k,v in self.data.items() if k!='events_json'};d.update(packed)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'a.npz';write_archive(p,d);loaded=read_archive(p)
            self.assertEqual(list(iter_events(loaded)),events)
    def test_legacy_pipeline(self):
        from tests.fixtures.legacy.PDE_to_ODE import generate
        from engine.numerics.ode import integrate
        from engine.visualization.archive import plot_archive
        with tempfile.TemporaryDirectory() as folder:
            source=generate({'dimensions':1,'mode_ranges':([0,1],),'fields':('u',),'parameters':{'D':.1,'Lx1':1},'rhs':{'u':'D*lap(u)'},'cache_dir':folder})
            data=integrate(source,{('u',(0,)):1,('u',(1,)):.2},{},0,.2,save_dt=.1)
            expected=.2*np.exp(-.1*np.pi**2*data['time'])
            np.testing.assert_allclose(data['coefficients'][:,1],expected,rtol=1e-6)
            p=Path(folder)/'old.npz';write_archive(p,data);read_archive(p)
            plot_archive(p,frame_index=2,save=Path(folder)/'old.png')
    def test_unmodified_analysis_scripts(self):
        for name in ('PDE_to_ODE.py','CHEYBYSHEV_approximation.py'):
            import hashlib
            hashes=json.loads((ROOT/'tests/fixtures/legacy/hashes.json').read_text())
            self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/legacy'/name).read_bytes()).hexdigest(),hashes[name])

if __name__=='__main__': unittest.main()
