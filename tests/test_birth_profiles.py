"""Spatial profile acceptance checks; hazard/parameter policies stay independent."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.events.base import PBirthEvent
from engine.galerkin.run import EventState, SegmentedRun, normalize_event
from engine.model.birth_profiles import evaluate_birth_profile, register_birth_profile, BIRTH_PROFILES
from engine.galerkin.basis import TensorBasis
from engine.archive import read_archive, decode
from test_p_birth import config, birth


class BirthProfileTests(unittest.TestCase):
    def state(self):
        s=EventState(config())
        s.u[0]=np.linspace(0,2,s.grid.shape[0])
        return s

    def apply(self,s,**params):
        e=normalize_event({'type':'p_birth','time':.1,**params},1)
        s.apply(e)
        return s.u[-1],e

    def test_uniform_omitted_equals_explicit_and_old_formula(self):
        a=self.state();b=self.state()
        a.accessibility[::3]=0.;b.accessibility[:]=a.accessibility
        old_rng=deepcopy(a.rng.bit_generator.state)
        ua,ea=self.apply(a,birth_amplitude=7)
        ub,eb=self.apply(b,birth_amplitude=7,birth_profile='uniform')
        np.testing.assert_array_equal(ua,7*a.accessibility)
        np.testing.assert_array_equal(ua,ub)
        self.assertEqual(a.registry,b.registry)
        self.assertEqual(a.rng.bit_generator.state,old_rng)
        self.assertEqual(ea.parameters['birth_profile'],'uniform')
        self.assertIsNone(ea.parameters['fallback_used'])
        self.assertEqual(ea.parameters['T_total_max'],2.)

    def test_inside_T_formula_active_family_sum_and_determinism(self):
        s=self.state();s.add_field('T_second','T',.5*s.u[0],0,{'diffusion':.001})
        s.add_field('P_existing','P',99,0,{'diffusion':.001})
        s.add_field('T_retired','T',100,0,{'diffusion':.001});s.retire(3,0)
        before=s.u.copy();rng=deepcopy(s.rng.bit_generator.state)
        p={'birth_profile':'inside_T','birth_profile_parameters':{'amplitude':1.7,'power':3.,'epsilon':1e-12}}
        u,a=evaluate_birth_profile(s.u,s.registry,s.grid,.1,p,s.accessibility)
        v,b=evaluate_birth_profile(s.u,s.registry,s.grid,.1,p,s.accessibility)
        total=before[0]+before[1]
        np.testing.assert_allclose(u,1.7*(total/(3.+1e-12))**3,atol=1e-15)
        np.testing.assert_array_equal(u,v);self.assertEqual(a,b)
        self.assertGreaterEqual(u.min(),0);self.assertAlmostEqual(u.max(),1.7,places=10)
        self.assertEqual(s.rng.bit_generator.state,rng);np.testing.assert_array_equal(s.u,before)
        self.assertEqual(a['T_total_max'],3.)

    def test_power_concentrates(self):
        s=self.state()
        values=[]
        for q in (1,4):
            u,_=evaluate_birth_profile(s.u,s.registry,s.grid,0,{'birth_profile':'inside_T','birth_profile_parameters':{'amplitude':2,'power':q}})
            values.append(u/u.max())
        self.assertTrue(np.all(values[1]<=values[0]+1e-14))
        self.assertLess(values[1][len(values[1])//2],values[0][len(values[0])//2])

    def test_zero_and_tiny_T_fallback_and_fail_transaction(self):
        for value in (0.,1e-14):
            s=self.state();s.u[0]=value
            u,e=self.apply(s,birth_profile='inside_T',birth_profile_parameters={'amplitude':2.})
            np.testing.assert_array_equal(u,2.)
            self.assertEqual(e.parameters['fallback_used'],'uniform');self.assertTrue(np.isfinite(u).all())
        s=self.state();s.u[:]=0;before=s.u.copy();rng=deepcopy(s.rng.bit_generator.state)
        with self.assertRaisesRegex(ValueError,'fallback=fail'):
            self.apply(s,birth_profile='inside_T',birth_profile_parameters={'fallback':'fail'},parameter_policy='random')
        np.testing.assert_array_equal(s.u,before);self.assertEqual(len(s.registry),1)
        self.assertEqual(s.rng.bit_generator.state,rng);self.assertEqual(s.events,[])

    def test_landscape_and_negative_spectral_values(self):
        s=self.state();s.u[0,0]=-.01;s.accessibility[4:12]=0;s.accessibility[12:16]=.3
        before=s.u.copy()
        u,e=self.apply(s,birth_profile='inside_T',birth_profile_parameters={'amplitude':1.,'power':2.5})
        self.assertEqual(u[0],0);np.testing.assert_array_equal(u[4:12],0.)
        np.testing.assert_allclose(u,np.maximum(before[0],0)**2.5/(2+1e-12)**2.5*s.accessibility,atol=1e-14)
        np.testing.assert_array_equal(s.u[0],before[0])
        self.assertEqual(e.parameters['negative_T_policy'],'clip_total_for_profile')

    def test_invalid_config_and_extensibility(self):
        s=self.state()
        for p in ({'power':0},{'power':-1},{'epsilon':0},{'amplitude':-1},{'amplitude':float('nan')},{'fallback':'skip'}):
            with self.subTest(p=p),self.assertRaises(ValueError):
                evaluate_birth_profile(s.u,s.registry,s.grid,0,{'birth_profile':'inside_T','birth_profile_parameters':p})
        with self.assertRaises(ValueError):evaluate_birth_profile(s.u,s.registry,s.grid,0,{'birth_profile':'unknown'})
        @register_birth_profile('test_profile')
        def custom(U,fields,grid,t,p):return np.full(grid.shape,p['amplitude']/2),p,None
        try:
            u,_=evaluate_birth_profile(s.u,s.registry,s.grid,0,{'birth_profile':'test_profile','birth_amplitude':4})
            np.testing.assert_array_equal(u,2)
        finally:del BIRTH_PROFILES['test_profile']

    def test_reproducibility_and_constant_hazard_uniform_comparison(self):
        c=config();c['initial_fields'][0]['initial']={'value':1,'geometry':{'type':'gaussian','center':[.5],'radius':.15}}
        c['stochastic_events']={'p_birth':birth(parameter_policy='inherit_mutate',fallback='random',
            parameter_ranges={'growth':[.01,.03],'capacity':[1,2]},mutation_sigma=.01,
            mobility_policy='inherit_mutate',diffusion_range=[.001,.002],mobility_mutation_sigma=.0001,
            birth_profile='inside_T',birth_profile_parameters={'amplitude':1,'power':3})}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            a=SegmentedRun(c,p/'a').run();b=SegmentedRun(c,p/'b').run()
            self.assertEqual(a['events'],b['events']);self.assertEqual(a['field_registry'],b['field_registry'])
            for x,y in zip(a['segments'],b['segments']):
                np.testing.assert_array_equal(read_archive(p/'a'/x['archive'])['coefficients'],read_archive(p/'b'/y['archive'])['coefficients'])
            plain=deepcopy(c);plain['stochastic_events']['p_birth']['birth_profile']='uniform'
            uniform=SegmentedRun(plain,p/'uniform').run()
            np.testing.assert_allclose([e['time'] for e in a['events']],[e['time'] for e in uniform['events']],atol=1e-12)
            for x,y in zip(a['events'],uniform['events']):
                self.assertEqual(x['parameters']['resulting_parameters'],y['parameters']['resulting_parameters'])
                # integrated_hazard is numerical solver output, not an RNG draw.
                self.assertAlmostEqual(x['rng']['integrated_hazard'],y['rng']['integrated_hazard'],places=12)
                self.assertEqual({k:v for k,v in x['rng'].items() if k!='integrated_hazard'},
                                 {k:v for k,v in y['rng'].items() if k!='integrated_hazard'})
            # The birth must follow T at event time, followed by existing projection.
            old=read_archive(p/'a'/a['segments'][0]['archive']);new=read_archive(p/'a'/a['segments'][1]['archive'])
            basis=TensorBasis.from_descriptor(decode(old,'basis_json'))
            T=basis.reconstruct(old['coefficients'][-1].reshape(basis.shape))
            profile=(np.maximum(T,0)/(np.maximum(T,0).max()+1e-12))**3
            expected=basis.project(profile).ravel()
            np.testing.assert_allclose(new['coefficients'][0,new['fields']=='P_0'],expected,atol=2e-13)
            self.assertGreater(np.ptp(profile),.5)
            for event in a['events']:
                for key in ('birth_profile','birth_profile_parameters','T_total_max','fallback_used'):
                    self.assertIn(key,event['parameters'])
                json.dumps(event,allow_nan=False)

    def test_2d_profile(self):
        c=config();c['dimension']=2;c['grid']={'shape':[12,14],'lengths':[2.,3.]}
        s=EventState(c);x,y=s.grid.mesh;s.u[0]=x*y
        u,a=evaluate_birth_profile(s.u,s.registry,s.grid,0,{'birth_profile':'inside_T'})
        self.assertEqual(u.shape,(12,14));self.assertGreater(np.ptp(u),1.)
