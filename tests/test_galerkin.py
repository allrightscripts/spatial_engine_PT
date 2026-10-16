import unittest
import numpy as np
from engine.galerkin.basis import TensorBasis
from engine.simulation.grid import Grid
from engine.model.loading import load_model

class BasisTests(unittest.TestCase):
    def test_project_reconstruct_all_bases_1d_2d(self):
        for dim in (1,2):
            for kind in ('cosine','sine','fourier'):
                grid=Grid([40]*dim,[1.]*dim)
                basis=TensorBasis(grid,[kind]*dim,[4]*dim)
                a=np.random.default_rng(5).normal(size=(2,)+basis.shape)
                np.testing.assert_allclose(basis.project(basis.reconstruct(a)),a,atol=5e-14)
    def test_eigenmodes_and_gradient(self):
        for kind in ('cosine','sine','fourier'):
            grid=Grid([64],[2.]);b=TensorBasis(grid,[kind],[4])
            a=np.zeros(b.shape);a[1]=1;k=b.frequencies[0][1];x=grid.coords[0]
            expected=np.sin(k*x) if kind=='sine' else np.cos(k*x)
            np.testing.assert_allclose(b.reconstruct(b.laplacian(a)),-k*k*expected,atol=1e-12)
            grad=k*np.cos(k*x) if kind=='sine' else -k*np.sin(k*x)
            np.testing.assert_allclose(b.gradient(a)[0],grad,atol=1e-12)
            np.testing.assert_allclose(b.weak_divergence(b.gradient(a)),b.laplacian(a),atol=1e-12)
    def test_mixed_tensor_basis(self):
        b=TensorBasis(Grid([32,40],[1,2]),['sine','cosine'],[3,4])
        a=np.ones(b.shape);np.testing.assert_allclose(b.project(b.reconstruct(a)),a,atol=1e-13)
    def test_two_thirds_has_no_dead_dofs(self):
        b=TensorBasis(Grid([24],[1]),['fourier'],[11],'2/3')
        self.assertEqual(b.shape,(15,));self.assertEqual(b.effective_modes,[7])
        a=np.zeros(b.shape);a[13]=1
        product=b.project(b.reconstruct(a)**2)
        self.assertAlmostEqual(product[0],.5)
        np.testing.assert_allclose(product[1:],0,atol=2e-14)
    def test_nonhomogeneous_rejected(self):
        with self.assertRaises(NotImplementedError):
            TensorBasis.from_config(Grid([20],[1]),[{'bc':{'type':'dirichlet','value':1}}],{'modes':[4]})
    def test_model_interface(self):
        m=load_model({'model':'cultural_v01'});g=Grid([8],[1]);u=np.ones((1,8))
        f=[{'parameters':{'diffusion':.1},'bc':{'type':'neumann'}}]
        np.testing.assert_allclose(m.rhs(u,f,g,0),0)

class ProviderTests(unittest.TestCase):
    def provider(self,kind='cosine'):
        from engine.galerkin.provider import GalerkinProvider
        bc={'cosine':'neumann','sine':'dirichlet','fourier':'periodic'}[kind]
        b=TensorBasis(Grid([32],[1]),[kind],[3])
        fields=[{'id':'u','family':'T','parameters':{'diffusion':.1},'bc':{'type':bc}}]
        return GalerkinProvider({'model':'cultural_v01'},fields,b)
    def test_all_integrators_and_archive(self):
        import tempfile
        from pathlib import Path
        from engine.numerics.ode import integrate_provider
        from engine.archive import write_archive,read_archive
        for method in ('RK45','DOP853','Radau','BDF','LSODA'):
            p=self.provider();a=np.array([1.,.2,0])
            d=integrate_provider(p,a,0,.1,method=method,save_dt=.05)
            np.testing.assert_allclose(d['coefficients'][-1],[1,.2*np.exp(-.1*np.pi**2*.1),0],atol=1e-7)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'segment.npz';write_archive(path,d);read_archive(path)
    def test_provider_analysis_adapter(self):
        import engine.numerics.newton as newton
        import engine.numerics.spectrum as spectrum
        from engine.numerics.rhs import evaluate_rhs
        p=self.provider('sine')
        for module in (newton,spectrum):
            order,params,compiled=module.load_equations(p.source())
            f=lambda a:evaluate_rhs(compiled,0,a,params)
            result=module.newton(f,np.ones(len(order)))
            np.testing.assert_allclose(result[0],0,atol=1e-7)
            J=module.numerical_jacobian(f,np.zeros(len(order)))
            np.testing.assert_allclose(np.sort(np.linalg.eigvals(J)),np.sort(-.1*p.basis.eigenvalues),atol=1e-8)

    def test_unversioned_original_archive(self):
        import tempfile
        from pathlib import Path
        from engine.numerics.ode import integrate
        from engine.archive import read_archive
        source="ORDER = [('u', (0,))]\nDEFAULT_PARAMETERS = {'Lx1': 1.0}\ndu_dt[0] = -u[0]\n"
        data=integrate(source,{('u',(0,)):1.},{},0,.1,save_dt=.1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'legacy.npz'
            np.savez_compressed(path,**data)
            restored=read_archive(path)
            self.assertNotIn('archive_schema_version',restored)
            np.testing.assert_allclose(restored['coefficients'][-1,0],np.exp(-.1),rtol=1e-6)
