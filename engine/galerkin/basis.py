"""Real tensor-product Fourier/sine/cosine Galerkin basis on midpoint grids.

Coefficients are physical amplitudes (not orthonormal transform coefficients).
Fourier slots: 0=1, 2k-1=cos(2πkx/L), 2k=sin(2πkx/L). Cosine indices start
at zero; sine indices at one. Precomputed axis matrices serve weak derivatives;
FFT/DCT/DST serve projection. No dense multi-dimensional basis is constructed.
"""
import numpy as np
from scipy.fft import dct, dst
from ..simulation.grid import Grid


def apply_axis(array, matrix, axis):
    """Apply matrix (output,input) along one tensor axis, retaining axis order."""
    result = np.tensordot(matrix, array, axes=(1, axis))
    return np.moveaxis(result, 0, axis)


class TensorBasis:
    # Validate per-axis basis/mode choices and precompute separable transforms,
    # derivatives and normalization. Fourier requests are frequency cutoffs, while
    # sine/cosine requests count basis functions.
    def __init__(self, grid, kinds, modes, dealiasing='none'):
        self.grid = grid
        self.kinds = list(kinds)
        self.requested_modes = list(modes)
        if len(kinds) != len(grid.shape) or len(modes) != len(kinds):
            raise ValueError('One basis and mode setting per spatial axis required')
        if dealiasing not in ('none', '2/3'):
            raise ValueError('dealiasing must be none or 2/3')
        self.dealiasing = dealiasing
        self.matrices, self.derivatives, self.norms = [], [], []
        self.indices, self.frequencies, self.effective_modes = [], [], []
        for kind, requested, n, x, L in zip(kinds, modes, grid.shape, grid.coords, grid.lengths):
            if type(requested) is not int or requested < 1:
                raise ValueError('Positive integer mode settings required')
            if kind == 'fourier':
                # Strict 2/3 cutoff eliminates quadratic aliasing into retained modes.
                # Discarded DOFs are not kept as fictitious zero eigenvalue directions.
                kmax = min(requested, (n-1)//3) if dealiasing == '2/3' else requested
                if 2*kmax >= n or kmax < 1:
                    raise ValueError('Fourier grid must resolve frequencies below Nyquist')
                k = np.repeat(np.arange(1, kmax+1), 2)
                freq = np.r_[0, k * (2*np.pi/L)]
                phase = x[:,None] * freq[None,:]
                B = np.cos(phase); D = -freq[None,:]*np.sin(phase)
                B[:,2::2] = np.sin(phase[:,2::2])
                D[:,2::2] = freq[None,2::2]*np.cos(phase[:,2::2])
                norm = np.r_[1.,np.full(2*kmax,.5)]
                index = np.arange(2*kmax+1)
                effective = kmax
            elif kind in ('sine','cosine'):
                if requested >= n:
                    raise ValueError('Sine/cosine grid needs more samples than modes')
                index = np.arange(requested) if kind == 'cosine' else np.arange(1, requested+1)
                freq = index*np.pi/L; phase = x[:,None]*freq[None,:]
                B = np.cos(phase) if kind == 'cosine' else np.sin(phase)
                D = -freq[None,:]*np.sin(phase) if kind == 'cosine' else freq[None,:]*np.cos(phase)
                norm = np.full(requested,.5)
                if kind == 'cosine': norm[0] = 1
                effective = requested
            else:
                raise ValueError(f'Unsupported basis {kind}')
            self.matrices.append(B); self.derivatives.append(D); self.norms.append(norm)
            self.indices.append(index); self.frequencies.append(freq); self.effective_modes.append(effective)
        self.shape = tuple(len(i) for i in self.indices)
        self.size = int(np.prod(self.shape))
        self.mode_indices = np.array([tuple(self.indices[a][i] for a,i in enumerate(idx))
                                      for idx in np.ndindex(self.shape)],dtype=np.int64)
        eigen = np.zeros(self.shape)
        for axis, freq in enumerate(self.frequencies):
            view = [1]*len(self.shape); view[axis] = len(freq)
            eigen += freq.reshape(view)**2
        self.eigenvalues = eigen
        # Geometry-only operators belong to the basis, not to each RHS call.
        self.projections = [B.T/n/norm[:,None] for B,n,norm in
                            zip(self.matrices,grid.shape,self.norms)]
        self.weak_projections = [D.T/n/norm[:,None] for D,n,norm in
                                 zip(self.derivatives,grid.shape,self.norms)]

    @classmethod
    def from_config(cls, grid, fields, settings):
        """Auto basis from common homogeneous BCs. No silent boundary lifting."""
        field_kinds = []
        for field in fields:
            kinds=[]
            for low,high in grid.boundaries(field['bc']):
                if low['type'] != high['type']:
                    raise NotImplementedError('Mixed low/high spectral BC needs another basis')
                if low.get('value',0) != 0 or high.get('value',0) != 0:
                    raise NotImplementedError('Nonhomogeneous spectral BC requires lifting; use finite volume')
                kinds.append({'periodic':'fourier','dirichlet':'sine','neumann':'cosine'}[low['type']])
            field_kinds.append(kinds)
        if not field_kinds or any(k != field_kinds[0] for k in field_kinds):
            raise NotImplementedError('v0.2 requires the same spectral basis for all fields')
        chosen=settings.get('basis','auto')
        if chosen != 'auto':
            chosen=[chosen]*len(grid.shape) if isinstance(chosen,str) else chosen
            if chosen != field_kinds[0]: raise ValueError('Explicit basis conflicts with BC')
        return cls(grid,field_kinds[0],settings['modes'],settings.get('dealiasing','none'))

    def project(self, values):
        """Tensor projection of trailing spatial axes, preserving leading batches."""
        a=np.asarray(values,float)
        if a.shape[-len(self.shape):] != self.grid.shape: raise ValueError('Projection grid shape mismatch')
        for axis,(kind,m,n) in enumerate(zip(self.kinds,self.shape,self.grid.shape)):
            ax=a.ndim-len(self.shape)+axis
            if kind == 'fourier':
                f=np.fft.rfft(a,axis=ax)/n
                k=np.arange((m-1)//2+1)
                phase=np.exp(-1j*np.pi*k/n)
                moved=np.moveaxis(f,ax,-1)[...,:len(k)]*phase
                out=np.empty(moved.shape[:-1]+(m,))
                out[...,0]=moved[...,0].real
                out[...,1::2]=2*moved[...,1:].real
                out[...,2::2]=-2*moved[...,1:].imag
                a=np.moveaxis(out,-1,ax)
            else:
                a=(dct(a,type=2,axis=ax) if kind=='cosine' else dst(a,type=2,axis=ax))/n
                a=np.take(a,np.arange(m),axis=ax)
                if kind=='cosine':
                    sl=[slice(None)]*a.ndim;sl[ax]=0;a[tuple(sl)]*=.5
        return a

    # Transform trailing mode axes into physical-grid axes, retaining leading field/batch
    # axes. Coefficients use this basis' amplitude normalization.
    def reconstruct(self, coefficients):
        a=np.asarray(coefficients,float)
        if a.shape[-len(self.shape):] != self.shape: raise ValueError('Coefficient shape mismatch')
        for axis,(kind,B,n) in enumerate(zip(self.kinds,self.matrices,self.grid.shape)):
            ax=a.ndim-len(self.shape)+axis
            if kind=='fourier':
                v=np.moveaxis(a,ax,-1);K=(v.shape[-1]-1)//2
                f=np.zeros(v.shape[:-1]+(n//2+1,),complex)
                f[...,0]=v[...,0]
                f[...,1:K+1]=(.5*(v[...,1::2]-1j*v[...,2::2]) * np.exp(1j*np.pi*np.arange(1,K+1)/n))
                a=np.moveaxis(np.fft.irfft(f,n=n,axis=-1)*n,-1,ax)
            else:
                a=apply_axis(a,B,ax)
        return a

    def derivative(self, coefficients, axis):
        """Exact derivative of the represented trigonometric polynomial at grid nodes."""
        a=np.asarray(coefficients,float)
        for j,B in enumerate(self.matrices):
            a=apply_axis(a,self.derivatives[j] if j==axis else B,a.ndim-len(self.shape)+j)
        return a

    # Reconstruct spatial derivatives along each physical axis using derivative matrices;
    # coordinates and lengths determine their units.
    def gradient(self, coefficients):
        return [self.derivative(coefficients,axis) for axis in range(len(self.shape))]

    def laplacian(self, coefficients):
        """Coefficient-space Laplacian; reconstruct its output for spatial values."""
        return -self.eigenvalues*np.asarray(coefficients)

    def weak_divergence(self, fluxes):
        """Project div(flux) = -<grad(phi),flux>/||phi||² (zero boundary term).

        Periodic cancellation, Dirichlet test functions zero, or natural zero
        Neumann flux are required. This avoids imposing an incorrect parity on
        fluxes when sine/cosine potentials are differentiated.
        """
        result=None
        for derivative_axis,flux in enumerate(fluxes):
            a=np.asarray(flux,float)
            for axis,(B,D,norm,n) in enumerate(zip(self.matrices,self.derivatives,self.norms,self.grid.shape)):
                projection=(self.weak_projections if axis==derivative_axis else self.projections)[axis]
                a=apply_axis(a,projection,a.ndim-len(self.shape)+axis)
            result=-a if result is None else result-a
        return result

    # Serialize grid, basis and truncation settings as ordinary data so archives can
    # reconstruct the same basis.
    def descriptor(self):
        return {'version':1,'kinds':self.kinds,'modes':self.requested_modes,
                'effective_modes':self.effective_modes,'dealiasing':self.dealiasing,
                'grid_shape':list(self.grid.shape),'lengths':list(self.grid.lengths),
                'normalization':'physical_amplitude','fourier_slots':'0=constant,2k-1=cos,2k=sin',
                'placement':'cell_center','tensor_order':'C'}

    # Rebuild the Grid and TensorBasis from stored settings; no model reaction code is
    # needed for reconstruction.
    @classmethod
    def from_descriptor(cls, descriptor):
        d=descriptor
        return cls(Grid(d['grid_shape'],d['lengths']),d['kinds'],d['modes'],d['dealiasing'])
