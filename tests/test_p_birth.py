from tests.support import config_path
"""Structural birth acceptance tests, hazard statistics and old-path guards."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.galerkin.run import SegmentedRun, EventState, normalize_event
from engine.galerkin.basis import TensorBasis
from engine.events.hazard import BirthClock
from engine.model.event_rules import RATES, evaluate_rate
from engine.model.mobility import validate_onsager
from engine.archive import read_archive, decode

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = json.loads((config_path('spectral_single.json')).read_text())
    c.update(t_end=.5, dt=.02, save_dt=.05, seed=2718)
    c['grid']['shape'] = [32]
    c['solver'].update(modes=[8], integrator='DOP853', max_step=.03, rtol=1e-9, atol=1e-11)
    c['initial_fields'][0]['parameters'] = {'diffusion':.001}
    c['split']['enabled'] = False
    return c


def birth(**updates):
    return {**{'enabled':True, 'max_births':2,
        'rate_parameters':{'lambda0':10., 'alpha':0},
        'birth_amplitude':10., 'parameter_policy':'manual',
        'field_parameters':{'growth':0., 'capacity':20.},
        'mobility_policy':'manual', 'diffusion':.001}, **updates}


def fields_at(data, index=0):
    b = TensorBasis.from_descriptor(decode(data, 'basis_json'))
    return b.reconstruct(data['coefficients'][index].reshape((-1,)+b.shape))


class BirthTests(unittest.TestCase):
    def test_constant_hazard_full_run_and_standard_archives(self):
        c = config(); c['stochastic_events'] = {'p_birth':birth()}
        with tempfile.TemporaryDirectory() as d:
            m = SegmentedRun(c,d).run()
            self.assertEqual([len(s['field_ids']) for s in m['segments']], [1,2,3])
            self.assertEqual([s['dof'] for s in m['segments']], [8,16,24])
            times = [e['time'] for e in m['events']]
            expected = np.cumsum(np.random.default_rng(c['seed']).exponential(size=2))/10
            np.testing.assert_allclose(times, expected, atol=1e-12)
            self.assertEqual([r['to'] for r in m['relations']], [1,2])
            for segment in m['segments']:
                a = read_archive(Path(d)/segment['archive'])
                self.assertEqual(a['coefficients'].shape[1],segment['dof'])
                self.assertEqual(float(a['time'][-1]),segment['end_time'])
                self.assertTrue(all(v.dtype != object for v in a.values()))
                self.assertTrue(np.isfinite(a['coefficients']).all())
            a,b,cdata = [read_archive(Path(d)/s['archive']) for s in m['segments']]
            np.testing.assert_allclose(fields_at(b)[-1],10,atol=1e-12)
            np.testing.assert_allclose(fields_at(cdata)[-1],10,atol=1e-12)
            np.testing.assert_allclose(fields_at(a,-1),fields_at(b)[:1],atol=1e-12)
            self.assertEqual(m['field_registry'][-1]['birth_segment'],2)
            self.assertEqual(m['field_registry'][-1]['model_type'],'progressive_v1')
            self.assertEqual(m['genealogy'][-1]['child_id'],'P_1')
            # Both universal visualization and Newton/spectrum provider interfaces.
            from engine.visualization.archive import plot_archive
            from engine.numerics.rhs import load_provider
            plot_archive(Path(d)/m['segments'][1]['archive'],frame_index=0,save=Path(d)/'birth.png')
            p=load_provider(str(b['equations_text'].item()))
            self.assertEqual(p.rhs(times[0],b['initial_coefficients']).shape,(16,))

    def test_manual_birth_uses_identical_map_and_exact_time(self):
        c=config();c['events']=[{'type':'p_birth','time':.137,'policy':'manual',
                                'birth_amplitude':7,'field_parameters':{'growth':.1},'diffusion':.02}]
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run()
            self.assertEqual(len(m['segments']),2)
            self.assertEqual(m['segments'][0]['end_time'],.137)
            e=m['events'][0];self.assertEqual(e['event_type'],'p_birth')
            self.assertEqual(e['parameters']['mobility_creation']['diffusion'],.02)
            a=read_archive(Path(d)/m['segments'][1]['archive'])
            np.testing.assert_allclose(fields_at(a)[-1],7,atol=1e-12)

    def test_reproducibility_random_inheritance_and_other_events(self):
        c=config();c['stochastic_events']={'p_birth':birth(parameter_policy='inherit_mutate',fallback='random',
            parameter_ranges={'growth':[.1,.3],'capacity':[12,20],'arbitrary':[0,1]},mutation_sigma=.05,
            parent_policy='weighted_by_mass',mobility_policy='inherit_mutate',diffusion_range=[.001,.004],
            mobility_mutation_sigma=.001)}
        c['stochastic_catastrophe']={'rate':4,'event':{'shape':'gaussian','radius':.1,'strength':.1,'targets':['*']}}
        with tempfile.TemporaryDirectory() as d:
            a=SegmentedRun(c,Path(d)/'a').run();b=SegmentedRun(c,Path(d)/'b').run()
            self.assertEqual(a['events'],b['events']);self.assertEqual(a['field_registry'],b['field_registry'])
            for x,y in zip(a['segments'],b['segments']):
                np.testing.assert_array_equal(read_archive(Path(d)/'a'/x['archive'])['coefficients'],
                                              read_archive(Path(d)/'b'/y['archive'])['coefficients'])
            self.assertEqual(a['field_registry'][-1]['parent_id'],'P_0')
            self.assertNotEqual(a['field_registry'][-1]['parameters'],a['field_registry'][-2]['parameters'])
            for f in a['field_registry'][1:]:
                for k,(lo,hi) in c['stochastic_events']['p_birth']['parameter_ranges'].items():
                    self.assertTrue(lo <= f['parameters'][k] <= hi)
                self.assertTrue(.001 <= f['parameters']['diffusion'] <= .004)
            c['seed']+=1;z=SegmentedRun(c,Path(d)/'z').run()
            self.assertNotEqual(a['events'],z['events'])

    def test_zero_mutation_inherits_and_preserves_mobility_block(self):
        c=config();s=EventState(c)
        s.apply(normalize_event({'type':'p_birth','time':.1,'field_parameters':{'growth':.7,'foo':4},
                                'diffusion':.02,'couplings':{'T_root':.001}},1))
        old=s.model.mobility.birth_matrix(s.u,s.registry,s.grid,.1).copy()
        parent=deepcopy(s.registry[-1]['parameters'])
        s.apply(normalize_event({'type':'p_birth','time':.2,'parameter_policy':'inherit_mutate',
            'mobility_policy':'inherit_mutate','mutation_sigma':0,'mobility_mutation_sigma':0},1))
        self.assertEqual(s.registry[-1]['parameters'],parent)
        self.assertEqual(s.registry[-1]['generation'],1)
        L=s.model.mobility.birth_matrix(s.u,s.registry,s.grid,.2)
        np.testing.assert_array_equal(L[:2,:2],old)
        self.assertEqual(L.shape,(3,3));validate_onsager(L)

    def test_bounds_clipping_and_invalid_onsager_transaction(self):
        s=EventState(config())
        s.apply(normalize_event({'type':'p_birth','time':.1,'field_parameters':{'foo':100},
                                'parameter_ranges':{'foo':[0,1]},'diffusion':.01},1))
        self.assertEqual(s.registry[-1]['parameters']['foo'],1)
        n=len(s.registry)
        with self.assertRaisesRegex(ValueError,'positive semidefinite'):
            s.apply(normalize_event({'type':'p_birth','time':.2,'diffusion':.001,
                                    'couplings':{'T_root':10}},1))
        self.assertEqual(len(s.registry),n)
        for L in ([[1,2],[0,1]],[[1,2],[2,1]]):
            with self.assertRaises(ValueError):validate_onsager(L)
        validate_onsager([[1,2],[0,1]],dissipative=False)

    def test_disabled_exact_trajectory_and_rng_compatibility(self):
        c=config()
        with tempfile.TemporaryDirectory() as d:
            a=SegmentedRun(c,Path(d)/'a').run()
            c['stochastic_events']={'p_birth':{'enabled':False}}
            b=SegmentedRun(c,Path(d)/'b').run()
            self.assertEqual(a['rng_state'],b['rng_state'])
            np.testing.assert_array_equal(read_archive(Path(d)/'a'/a['segments'][0]['archive'])['coefficients'],
                                          read_archive(Path(d)/'b'/b['segments'][0]['archive'])['coefficients'])

    def test_time_and_state_dependent_rate_without_solver_changes(self):
        RATES['test_time'] = lambda U,fields,grid,t,params: 2*t
        c=config();c['t_end']=1;c['stochastic_events']={'p_birth':birth(max_births=1,rate_function='test_time')}
        try:
            with tempfile.TemporaryDirectory() as d:
                m=SegmentedRun(c,d).run()
                self.assertAlmostEqual(m['events'][0]['time'],np.sqrt(np.random.default_rng(c['seed']).exponential()),places=9)
        finally:del RATES['test_time']
        c=config();c['t_end']=2;c['stochastic_events']={'p_birth':birth(rate_parameters={'lambda0':1,'alpha':1})}
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run();e0,e1=m['events']
            draws=np.random.default_rng(c['seed']).exponential(size=2)
            self.assertAlmostEqual(e0['time'],draws[0],places=9)
            self.assertAlmostEqual(e1['time']-e0['time'],draws[1]/11,places=9)

    def test_clock_survives_unrelated_segment_changes(self):
        c=config();c['stochastic_events']={'p_birth':birth(max_births=1)}
        c['events']=[{'type':'catastrophe','time':.007,'targets':['T_root'],'strength':.1}]
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run()
            self.assertAlmostEqual(m['events'][1]['time'],np.random.default_rng(c['seed']).exponential()/10,places=10)

    def test_rate_validation_and_model_dispatch(self):
        s=EventState(config())
        with self.assertRaises(ValueError):evaluate_rate('default_p_birth_rate',s.u,s.registry,s.grid,0,{'lambda0':-1})
        from engine.model.cultural_v01 import REACTION_LAWS
        REACTION_LAWS['progressive_test']=lambda i,U,fields,grid,t,total:np.full(grid.shape,3.)
        try:
            s.apply(normalize_event({'type':'p_birth','time':0,'model_type':'progressive_test'},1))
            R=s.model.reaction(s.u,s.registry,s.grid,0)
            np.testing.assert_array_equal(R[1],3)
            self.assertTrue(np.all(R[0]==0))
        finally:del REACTION_LAWS['progressive_test']

    def test_waiting_time_statistics_and_cadence_refinement(self):
        # Ensemble tolerance ~5 standard errors; deterministic seed set, no flaky
        # p-value gate. Testing the exact Exp clock plus its integration path.
        waits=np.array([BirthClock(birth(),np.random.default_rng(i),i).threshold/2 for i in range(2048)])
        self.assertLess(abs(waits.mean()-.5),.06)
        self.assertLess(abs(np.mean(waits <= .5)-(1-np.exp(-1))),.05)
        from engine.numerics.ode import integrate_provider
        from engine.galerkin.provider import GalerkinProvider
        s=EventState(config());basis=TensorBasis(s.grid,['cosine'],[2]);p=GalerkinProvider(config(),s.registry,basis)
        a=basis.project(s.u).ravel()
        results=[]
        for step in (.2,.02):
            times=[]
            for seed in range(24):
                E=float(np.random.default_rng(seed).exponential())
                result=integrate_provider(p,a,0,E/2+.1,method='DOP853',max_step=step,
                    integral_clock={'rate':lambda t,a:2.,'initial':0.,'threshold':E})
                times.append(float(result['time'][-1]))
                self.assertAlmostEqual(times[-1],E/2,places=10)
            results.append(times)
        np.testing.assert_allclose(*results,atol=1e-10)

    def test_hazard_supported_solve_ivp_methods(self):
        c=config();c['stochastic_events']={'p_birth':birth(max_births=1)}
        for method in ('RK45','DOP853','Radau','BDF','LSODA'):
            c['solver']['integrator']=method
            with tempfile.TemporaryDirectory() as d:
                m=SegmentedRun(c,d).run()
                self.assertAlmostEqual(m['events'][0]['time'],.014828003311570765,places=9)

    def test_random_cross_couplings_and_inherited_full_row(self):
        c=config();s=EventState(c)
        s.apply(normalize_event({'type':'p_birth','time':.1,'parameter_policy':'random',
            'parameter_ranges':{'foo':[-2,3]},'mobility_policy':'random','diffusion_range':[.01,.02],
            'cross_couplings':True,'coupling_range':[.0001,.0002]},1))
        old=s.model.mobility.birth_matrix(s.u,s.registry,s.grid,.1).copy()
        self.assertTrue(.0001<=old[0,1]<=.0002)
        self.assertTrue(.01<=old[1,1]<=.02)
        s.apply(normalize_event({'type':'p_birth','time':.2,'parameter_policy':'inherit_mutate',
            'mobility_policy':'inherit_mutate','cross_couplings':True,'mutation_sigma':0,
            'mobility_mutation_sigma':0},1))
        new=s.model.mobility.birth_matrix(s.u,s.registry,s.grid,.2)
        np.testing.assert_array_equal(new[:2,:2],old)
        np.testing.assert_array_equal(new[-1,:2],old[-1])
        validate_onsager(new)

    def test_birth_2d_and_periodic_without_special_indices(self):
        c=config();c['dimension']=2;c['grid']={'shape':[16,18],'lengths':[1,2]}
        c['boundary']={'type':'periodic'};c['solver'].update(modes=[2,2],dealiasing='2/3')
        c['events']=[{'time':.1,'type':'p_birth'}]
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run();a=read_archive(Path(d)/m['segments'][1]['archive'])
            np.testing.assert_allclose(fields_at(a)[-1],10,atol=1e-12)
            self.assertEqual(a['coefficients'].shape[1],50)

    def test_manual_at_zero_and_final_time(self):
        c=config();c['events']=[{'time':0,'type':'p_birth'},{'time':c['t_end'],'type':'p_birth'}]
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run()
            self.assertEqual(len(m['segments']),1)
            self.assertEqual(m['field_registry'][1]['birth_segment'],0)
            self.assertEqual(m['relations'][-1]['to'],None)
            with np.load(Path(d)/m['terminal_state'],allow_pickle=False) as a:
                self.assertEqual(len(a['fields']),3)
                np.testing.assert_array_equal(a['values'][-1],10)

    def test_clock_metadata_is_final_not_first_control_slab(self):
        c=config();c['stochastic_events']={'p_birth':birth(rate_parameters={'lambda0':.01})}
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run();a=read_archive(Path(d)/m['segments'][0]['archive'])
            self.assertAlmostEqual(decode(a,'metadata_json')['integral_clock']['value'],.005,places=12)
