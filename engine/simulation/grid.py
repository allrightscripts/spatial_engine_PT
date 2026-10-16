"""Cell-centred conservative fluxes; no Python loops over spatial cells.

Boundary values apply to chemical potentials mu, i.e. the transported
variables. For the demo mu=U, these are also the physical field BCs.
Neumann values are OUTWARD normal derivatives on both sides.
"""
import numpy as np

class Grid:
    # Validate 1D/2D/3D shape and positive lengths, then construct cell centres, spacings
    # and one-cell volume used in integrals.
    def __init__(self, shape, lengths):
        self.shape = tuple(shape); self.lengths = tuple(lengths)
        if len(shape) not in (1,2,3) or len(lengths) != len(shape):
            raise ValueError('Only 1D/2D/3D grids supported')
        if any(type(n) is not int or n < 2 for n in shape) or any(not np.isfinite(x) or x <= 0 for x in lengths):
            raise ValueError('Invalid grid')
        self.dx = np.array(lengths)/shape
        self.volume = float(np.prod(self.dx))
        self.coords = [(np.arange(n)+.5)*h for n,h in zip(shape,self.dx)]
        self.mesh = np.meshgrid(*self.coords,indexing='ij')

    def boundaries(self, spec):
        """Expand one BC or per-axis low/high pairs; periodic must be paired."""
        axes = spec.get('axes', [[spec,spec] for _ in self.shape])
        if len(axes) != len(self.shape): raise ValueError('BC axis count')
        for pair in axes:
            if len(pair) != 2: raise ValueError('BC needs low/high pair')
            kinds = [s['type'] for s in pair]
            if any(k not in ('periodic','dirichlet','neumann') for k in kinds): raise ValueError('Unknown BC')
            if ('periodic' in kinds) and kinds != ['periodic','periodic']: raise ValueError('Unpaired periodic BC')
            if any(not np.isfinite(s.get('value',0)) for s in pair): raise ValueError('Invalid BC value')
        return axes

    def gradient_faces(self, u, bc, axis):
        """Interior differences and exact half-cell Dirichlet distances."""
        v = np.moveaxis(u,axis,0); h = self.dx[axis]
        out = np.empty((v.shape[0]+1,)+v.shape[1:])
        out[1:-1] = np.diff(v,axis=0)/h
        low, high = self.boundaries(bc)[axis]
        if low['type'] == 'periodic':
            out[0] = out[-1] = (v[0]-v[-1])/h
        else:
            out[0] = (v[0]-low.get('value',0))/(h/2) if low['type']=='dirichlet' else -low.get('value',0)
            out[-1] = (high.get('value',0)-v[-1])/(h/2) if high['type']=='dirichlet' else high.get('value',0)
        return np.moveaxis(out,0,axis)

    # Compute the two-cell arithmetic mean used to place cell values on shared faces;
    # preserves NumPy broadcasting.
    def face_average(self, a, axis, periodic):
        v = np.moveaxis(a,axis,0)
        out = np.empty((len(v)+1,)+v.shape[1:]); out[1:-1]=(v[1:]+v[:-1])/2
        out[0],out[-1] = v[0],v[-1]
        if periodic: out[0]=out[-1]=(v[0]+v[-1])/2
        return np.moveaxis(out,0,axis)

    def permeability_faces(self, permeability, axis, is_periodic):
        """Harmonic face coefficient: a zero cell blocks both adjoining faces."""
        v=np.moveaxis(permeability,axis,0)
        face=np.empty((len(v)+1,)+v.shape[1:])
        face[1:-1]=np.divide(2*v[:-1]*v[1:],v[:-1]+v[1:],
            out=np.zeros_like(v[:-1]),where=(v[:-1]+v[1:])>0)
        face[0],face[-1]=v[0],v[-1]
        if is_periodic:
            face[0]=face[-1]=np.divide(2*v[0]*v[-1],v[0]+v[-1],
                out=np.zeros_like(v[0]),where=(v[0]+v[-1])>0)
        return np.moveaxis(face,0,axis)

    def transport(self, mu, mobility, bcs, permeability=None):
        """Compute div(L grad(mu)); L is (N,N,*shape), may depend on U.

        Mixed topologies with off-diagonal transport are deliberately rejected:
        a cross-flux cannot be periodic for one field and open for another.
        """
        if permeability is not None:
            permeability=np.broadcast_to(np.asarray(permeability,float),self.shape)
            if not np.isfinite(permeability).all() or np.any((permeability<0)|(permeability>1)):
                raise ValueError('Permeability must be finite in [0,1]')
        n = len(mu); result = np.zeros_like(mu)
        L = np.broadcast_to(mobility,(n,n)+self.shape)
        axes_bc = [self.boundaries(bc) for bc in bcs]
        for axis,h in enumerate(self.dx):
            grad = [self.gradient_faces(mu[j],bcs[j],axis) for j in range(n)]
            periodic = [b[axis][0]['type']=='periodic' for b in axes_bc]
            permeability_faces=({p:self.permeability_faces(permeability,axis,p) for p in set(periodic)}
                                if permeability is not None else {})
            for i in range(n):
                flux = np.zeros_like(grad[0])
                for j in range(n):
                    if not np.any(L[i,j]): continue
                    if periodic[i] != periodic[j]: raise ValueError('Cross mobility requires compatible periodic topology')
                    flux += self.face_average(L[i,j],axis,periodic[i])*grad[j]
                if permeability is not None:
                    flux *= permeability_faces[periodic[i]]
                result[i] += np.diff(flux,axis=axis)/h
        return result
