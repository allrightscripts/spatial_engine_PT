"""Independent functional, mobility and reaction components for a demo model."""
import numpy as np

class QuadraticFunctional:
    """F=1/2 integral sum U_i^2; variational derivative is mu_i=U_i."""
    # Return mu=U, the variational derivative of the quadratic functional. The returned
    # array aliases the input: downstream code must not mutate it.
    def chemical_potential(self,u,fields,grid,t): return u
    # Compute one half of the sum of squared field values times cell volume, returning a
    # scalar functional value.
    def value(self,u,fields,grid,t): return float(.5*np.sum(u*u)*grid.volume)

class DiagonalMobility:
    # Construct a diagonal field mobility tensor from each field's diffusion parameter;
    # singleton spatial axes enable broadcasting.
    def __call__(self,u,fields,grid,t):
        n=len(fields); L=np.zeros((n,n)+(1,)*len(grid.shape))
        for i,f in enumerate(fields): L[i,i]=f['parameters'].get('diffusion',0)
        return L

class LogisticCompetition:
    """Local growth and competition; no family-specific logic in the solver."""
    # Evaluate local self-limited growth, cross-field competition and decay for every
    # field; return an array with the same shape as u.
    def __call__(self,u,fields,grid,t):
        total=u.sum(axis=0); out=np.empty_like(u)
        for i,f in enumerate(fields):
            p=f['parameters']
            out[i]=p.get('growth',0)*u[i]*(1-u[i]/p.get('capacity',1))-p.get('competition',0)*u[i]*(total-u[i])-p.get('decay',0)*u[i]
        return out

# A new model module should expose build_model(parameters) -> Model.
# Functional chemical_potential and reaction must return arrays shaped like u;
# mobility must follow grid.transport's field-by-field tensor convention.
# See docs/EXTENDING.md before changing the three components independently.
class Model:
    """Inject any three compatible callables; archive code needs no change."""
    # Compose functional, mobility and reaction components, using demo defaults for
    # omitted components. Custom components must follow their array contracts.
    def __init__(self,functional=None,mobility=None,reaction=None):
        self.functional=functional or QuadraticFunctional()
        self.mobility=mobility or DiagonalMobility()
        self.reaction=reaction or LogisticCompetition()
    # Combine transport of chemical potentials with local reactions on the grid. This
    # generic FV path does not contain cultural family rules.
    def rhs(self,u,fields,grid,t):
        mu=self.functional.chemical_potential(u,fields,grid,t)
        return grid.transport(mu,self.mobility(u,fields,grid,t),[f['bc'] for f in fields])+self.reaction(u,fields,grid,t)
