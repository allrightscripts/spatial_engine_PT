"""Numerical guards for operator shortcuts and continuation at control boundaries."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
from engine.galerkin.integration import IntegrationSession
from engine.galerkin.provider import GalerkinProvider
from engine.galerkin.basis import TensorBasis
from engine.simulation.grid import Grid


class PerformanceTests(unittest.TestCase):
    def test_transport_matches_weak_form_for_edited_laws(self):
        rng = np.random.default_rng(83)
        for kinds in (['cosine'], ['sine'], ['fourier'],
                      ['cosine','sine'], ['fourier','cosine'], ['sine','sine']):
            grid = Grid([24]*len(kinds), [1.3]*len(kinds))
            b = TensorBasis(grid, kinds, [4]*len(kinds))
            fields = [{'id':str(i), 'parameters':{}} for i in range(2)]
            state = rng.normal(scale=.02, size=2*b.size)
            for spatial, barrier in ((False,False),(True,False),(False,True)):
                with self.subTest(kinds=kinds, spatial=spatial, barrier=barrier):
                    access = .6+.2*np.cos(grid.mesh[0]) if barrier else None
                    p = GalerkinProvider({'model':'cultural_v01'},fields,b,accessibility=access)
                    def mobility(U, fields, grid, t):
                        L = np.array([[.2,.04],[.04,.3]])*(1+t+U.mean()**2)
                        L = L.reshape((2,2)+(1,)*len(kinds))
                        return L*(1+.1*grid.mesh[0]) if spatial else L
                    p.model = SimpleNamespace(
                        functional=SimpleNamespace(chemical_potential=lambda U,f,g,t: U+U**3+t),
                        mobility=mobility, reaction=lambda U,f,g,t: .2*U-U**2)
                    t = .37
                    U = p.physical_state(state)
                    mu = p.model.functional.chemical_potential(U,fields,grid,t)
                    L = np.broadcast_to(mobility(U,fields,grid,t),(2,2)+grid.shape)
                    reaction = p.model.reaction(U,fields,grid,t)
                    if access is not None:
                        L = L*access
                        reaction = reaction*access
                    flux = [np.einsum('ij...,j...->i...',L,g) for g in b.gradient(b.project(mu))]
                    reference = b.weak_divergence(flux)+b.project(reaction)
                    np.testing.assert_allclose(p.rhs(t,state),reference.ravel(),rtol=2e-12,atol=2e-12)

    def test_session_nonlinear_stiff_time_dependent_and_root(self):
        for method in IntegrationSession.methods:
            for threshold in (None, .73):
                with self.subTest(method=method,threshold=threshold):
                    def rhs(t,y):
                        return np.array([-40*(y[0]-np.cos(t))-np.sin(t)+.1*y[1]**2,
                                         -.4*y[1]**2, 1+y[0]**2])
                    event = None if threshold is None else lambda t,y:y[2]-threshold
                    if event is not None:
                        event.terminal=True
                        event.direction=1
                    settings=dict(method=method,rtol=1e-9,atol=1e-11,max_step=.08)
                    reference=solve_ivp(rhs,(0,1),[1,.7,0],dense_output=True,events=event,**settings)
                    session=IntegrationSession()
                    initial=np.array([1,.7,0.])
                    start=0.
                    for end in (.11,.29,.53,.71,1.):
                        times=np.linspace(start,end,7)
                        result=session.solve(rhs,(start,end),initial,t_eval=times,events=event,**settings)
                        self.assertTrue(result.success)
                        np.testing.assert_allclose(result.y,reference.sol(result.t),rtol=2e-8,atol=2e-9)
                        if len(result.t_events[0]):
                            self.assertAlmostEqual(result.t_events[0][0],reference.t_events[0][0],places=8)
                            self.assertIsNone(session.solver)
                            break
                        initial=result.y[:,-1]
                        start=end
                    else:
                        self.assertIsNone(threshold)

    def test_resume_rejects_mutated_state(self):
        s=IntegrationSession()
        kwargs=dict(method='Radau',rtol=1e-7,atol=1e-9,max_step=.1)
        a=s.solve(lambda t,y:-y,(0,.2),np.ones(1),t_eval=[0,.2],**kwargs)
        with self.assertRaises(ValueError):
            s.solve(lambda t,y:-y,(.2,.4),a.y[:,-1]+1,t_eval=[.2,.4],**kwargs)
