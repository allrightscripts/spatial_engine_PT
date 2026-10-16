"""Custom masks, operator changes and quantified hard-barrier discretization."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from engine.model.masks import MASKS, evaluate_mask
from engine.events.base import LandscapeEvent, CatastropheEvent
from engine.galerkin.run import EventState, SegmentedRun, normalize_event
from engine.galerkin.provider import GalerkinProvider, from_descriptor
from engine.galerkin.basis import TensorBasis
from engine.simulation.grid import Grid
from engine.archive import read_archive, decode
from test_p_birth import config, fields_at


class LandscapeTests(unittest.TestCase):
    def test_all_builtin_masks_1d_2d_and_custom(self):
        for dimension in (1,2):
            g=Grid([32]*dimension,[1.]*dimension)
            for name in ('interval','rectangle','circle','ellipse','gaussian'):
                m=evaluate_mask(g,{'type':name,'low':[.3]*dimension,'high':[.7]*dimension,
                                   'center':[.5]*dimension,'radii':[.2]*dimension})
                self.assertEqual(m.shape,g.shape);self.assertTrue(np.all((m>=0)&(m<=1)))
            custom=evaluate_mask(g,{'mask':'my_catastrophe','parameters':{'a':.5,'A':.2,'b':.5,'B':.1}})
            expected=((g.mesh[0]-.5)/.2)**2
            if dimension==2:expected+=((g.mesh[1]-.5)/.1)**2
            np.testing.assert_array_equal(custom,expected<=1)

    def test_custom_catastrophe_time_sampling_and_rate_snapshot(self):
        MASKS['moving_test']=lambda X,*,t,params:X<t
        try:
            s=EventState(config());s.u[:]=1
            s.apply(normalize_event({'type':'catastrophe','time':.3,'mask':'moving_test',
                                    'targets':['T_root'],'strength':1},1))
            np.testing.assert_array_equal(s.u[0],s.grid.coords[0]>=.3)
            s.apply(normalize_event({'type':'catastrophe','time':.3,'mask':'moving_test',
                                    'targets':['T_root'],'strength':2,'duration':.1},1))
            b=TensorBasis(s.grid,['cosine'],[8]);p=GalerkinProvider(config(),s.registry,b,s.active_forcing())
            np.testing.assert_array_equal(p.sinks[0][1],s.grid.coords[0]<.3)
            # Descriptor remains usable without re-evaluating the custom callable.
            del MASKS['moving_test']
            restored=from_descriptor(p.descriptor())
            np.testing.assert_array_equal(restored.sinks[0][1],p.sinks[0][1])
        finally:MASKS.pop('moving_test',None)

    def test_landscape_does_not_erase_fields_and_overlapping_temporary_layers(self):
        s=EventState(config());before=s.u.copy()
        s.apply(LandscapeEvent(.1,geometry={'type':'interval','low':[.4],'high':[.6]},
                               strength=1,parameters={'duration':.2}))
        first=s.events[-1]['event_id']
        s.apply(LandscapeEvent(.2,geometry={'type':'interval','low':[.5],'high':[.8]},strength=.5))
        s.apply(LandscapeEvent(.3,parameters={'phase':'end','start_event_id':first}))
        expected=np.where((s.grid.coords[0]>=.5)&(s.grid.coords[0]<=.8),.5,1)
        np.testing.assert_array_equal(s.accessibility,expected)
        np.testing.assert_array_equal(s.u,before)

    def test_operator_segments_and_birth_accessibility(self):
        c=config();c['events']=[
            {'time':.1,'type':'landscape','shape':'interval','low':[.4],'high':[.6],
             'strength':1,'duration':.2},
            {'time':.15,'type':'p_birth','birth_amplitude':10},
            {'time':.4,'type':'catastrophe','mask':'my_catastrophe','mask_parameters':{'A':.1},
             'targets':['P_0'],'strength':.5}]
        with tempfile.TemporaryDirectory() as d:
            m=SegmentedRun(c,d).run();self.assertEqual(len(m['segments']),5)
            self.assertEqual(m['segments'][0]['closure_reasons'],['landscape'])
            self.assertEqual(m['segments'][2]['closure_reasons'],['landscape_end'])
            for segment in m['segments']:read_archive(Path(d)/segment['archive'])
            a=read_archive(Path(d)/m['segments'][2]['archive'])
            b=TensorBasis.from_descriptor(decode(a,'basis_json'))
            np.testing.assert_allclose(a['initial_coefficients'][-b.size:],b.project(10*a['accessibility']),atol=1e-13)
            self.assertIn('l2_error',m['segments'][2]['projection_in'])
            p=from_descriptor(decode(a,'rhs_provider_json'))
            np.testing.assert_array_equal(p.accessibility,a['accessibility'])
        # Exact event-space initial condition before the documented truncation.
        s=EventState(c)
        s.apply(normalize_event(c['events'][0],1));s.apply(normalize_event(c['events'][1],1))
        np.testing.assert_array_equal(s.u[-1],10*s.accessibility)
        self.assertTrue(np.all(s.u[-1][s.accessibility==0]==0))

    def test_zero_landscape_disables_reaction_and_transport(self):
        c=config();s=EventState(c);s.registry[0]['parameters']['growth']=.5
        b=TensorBasis(s.grid,['cosine'],[8]);p=GalerkinProvider(c,s.registry,b,accessibility=np.zeros(s.grid.shape))
        np.testing.assert_array_equal(p.rhs(0,np.ones(b.size)),0)

    def test_finite_volume_hard_barrier_zero_transfer_1d_2d(self):
        for dimension in (1,2):
            g=Grid([24]*dimension,[1.]*dimension);x=g.mesh[0]
            M=np.where((x>.4)&(x<.6),0.,1.)
            initial=(x<.4).astype(float)[None]
            def rhs(t,a):
                return g.transport(a.reshape(initial.shape),np.array([[.03]]).reshape((1,1)+(1,)*dimension),
                                   [{'type':'neumann'}],permeability=M).ravel()
            sol=solve_ivp(rhs,(0,1),initial.ravel(),method='DOP853',rtol=1e-9,atol=1e-12)
            final=sol.y[:,-1].reshape(initial.shape)[0]
            self.assertLess(np.max(np.abs(final[x>.6])),1e-12)
            self.assertAlmostEqual(final.sum(),initial.sum(),places=10)

    def test_spectral_barrier_quantified_truncation_error(self):
        # A global finite trigonometric expansion cannot impose exact local
        # support. Check absolute leaked L1 mass (<1.5e-4), refinement and an
        # unobstructed control. This is a numerical tolerance, not an exact claim.
        g=Grid([384],[1]);x=g.coords[0];M=np.where((x>.4)&(x<.6),0.,1.)
        z=(x-.23)/.15;u=np.zeros(g.shape);inside=np.abs(z)<1
        u[inside]=np.exp(-1/(1-z[inside]**2))
        fields=[{'id':'T','family':'T','parameters':{'diffusion':.02},'bc':{'type':'neumann'}}]
        leaks=[]
        for modes,mask in ((64,M),(96,M),(96,np.ones_like(M))):
            b=TensorBasis(g,['cosine'],[modes]);p=GalerkinProvider({'model':'cultural_v01'},fields,b,accessibility=mask)
            a=b.project(u);sol=solve_ivp(p.rhs,(0,2),a,method='BDF',rtol=1e-9,atol=1e-11)
            self.assertTrue(sol.success)
            final=b.reconstruct(sol.y[:,-1])
            leaks.append(float(np.abs(final[x>.6]).sum()*g.volume))
            self.assertAlmostEqual(final.sum()*g.volume,u.sum()*g.volume,places=10)
        self.assertLess(leaks[1],1.5e-4)
        self.assertLess(leaks[1],leaks[0])
        self.assertLess(leaks[1],.03*leaks[2])
