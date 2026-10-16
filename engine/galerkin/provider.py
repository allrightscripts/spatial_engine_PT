"""A fixed-dimensional, reproducible callable ODE provider for one segment."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
from .basis import TensorBasis
from ..model.loading import load_model
from ..events.operators import spatial_mask

FORMAT='galerkin_rhs_provider_v1'

# U is a physical grid array; a is a flattened vector of Galerkin coefficients.
# Reconstruction and projection translate between them. Preserve self.order:
# archive columns and post-processing depend on these stable (field ID, mode) pairs.
class GalerkinProvider:
    # Freeze active fields, basis, forcing and accessibility for one fixed-dimensional
    # ODE. Build stable coefficient ordering and validate compatible mobility.
    def __init__(self,config,fields,basis,forcing=None, accessibility=None):
        self.config=deepcopy(config);self.fields=deepcopy(fields);self.basis=basis
        self.model=load_model(config);self.grid=basis.grid
        self.forcing=deepcopy(forcing or [])
        self.accessibility=None if accessibility is None else np.asarray(accessibility,float).copy()
        if self.accessibility is not None:
            if self.accessibility.shape != self.grid.shape or not np.isfinite(self.accessibility).all() or np.any((self.accessibility<0)|(self.accessibility>1)):
                raise ValueError('Invalid accessibility field')
        self.representation='cosine_galerkin' if all(k=='cosine' for k in basis.kinds) else 'tensor_galerkin'
        self.order=[(f['id'],tuple(map(int,k))) for f in fields for k in basis.mode_indices]
        self.parameters={f'Lx{i+1}':v for i,v in enumerate(self.grid.lengths)}
        if 'mobility_validation' in config and hasattr(self.model.mobility,'birth_matrix'):
            from ..model.mobility import validate_onsager
            validate_onsager(self.model.mobility.birth_matrix(np.zeros((len(self.fields),)+self.grid.shape),self.fields,self.grid,0),
                config.get('mobility_validation',{}).get('onsager',getattr(self.model.mobility,'dissipative_onsager',False)))
        self.sinks=[]
        for force in self.forcing:
            mask=np.asarray(force['mask_values'],float) if 'mask_values' in force else spatial_mask(self.grid,force['geometry'])
            selected=np.array([f['id'] in force['targets'] for f in self.fields],float)
            self.sinks.append((selected.reshape((-1,)+(1,)*len(self.grid.shape)),mask,force['strength']))

    def physical_state(self, state):
        """Reuse reconstruction between RHS and deterministic hazard evaluation.

        The last coefficient array is copied because solve_ivp may reuse its
        buffers. Only this most recent evaluation is cached, never a trajectory.
        """
        state=np.asarray(state)
        if not hasattr(self,'_physical_a') or not np.array_equal(state,self._physical_a):
            self._physical_a=state.copy()
            self._physical_U=self.basis.reconstruct(state.reshape((len(self.fields),)+self.basis.shape))
        return self._physical_U

    def rhs(self,t,state):
        """Project chemical potential and reaction; weak divergence of L grad(mu).

        All work stays within this segment's fixed field/mode ordering. No random
        draws or detector updates occur inside an adaptive solver trial RHS.
        """
        b=self.basis
        U=self.physical_state(state)
        mu=self.model.functional.chemical_potential(U,self.fields,self.grid,t)
        mu_coefficients=b.project(mu)
        # Inspect the actual returned shape on every call: edited/time-dependent
        # laws stay live. Singleton spatial axes explicitly mean uniform L.
        L=np.asarray(self.model.mobility(U,self.fields,self.grid,t),float)
        full_shape=(len(self.fields),len(self.fields))+self.grid.shape
        spatially_uniform=(L.ndim == len(full_shape) and
                           all(n == 1 for n in L.shape[2:]))
        if spatially_uniform and self.accessibility is None:
            matrix=np.broadcast_to(L,full_shape[:2]+(1,)*len(self.grid.shape))
            matrix=matrix.reshape(full_shape[:2])
            # Exact same weak operator for this tensor basis, including cross
            # couplings; no assumption about the chemical potential's formula.
            transport=np.einsum('ij,j...->i...',matrix,b.laplacian(mu_coefficients))
        else:
            L=np.broadcast_to(L,full_shape)
            if self.accessibility is not None: L=L*self.accessibility
            grad=b.gradient(mu_coefficients)
            flux=[np.einsum('ij...,j...->i...',L,g,optimize=True) for g in grad]
            transport=b.weak_divergence(flux)
        reaction=self.model.reaction(U,self.fields,self.grid,t)
        if self.accessibility is not None: reaction=reaction*self.accessibility
        result=transport+b.project(reaction)
        for selected,mask,strength in self.sinks:
            result -= b.project(strength*selected*mask*U)
        return result.ravel()

    # Describe the frozen RHS as JSON-compatible data, including a byte hash of the model
    # source. Even model comments affect that audit hash.
    def descriptor(self):
        module=importlib.import_module('engine.model.'+self.config.get('model','cultural_v01'))
        source=Path(module.__file__).read_bytes()
        return {'format':FORMAT,'config':self.config,'fields':self.fields,
                'basis':self.basis.descriptor(),'forcing':self.forcing,
                **({'accessibility':self.accessibility.tolist()} if self.accessibility is not None else {}),
                'model_source_sha256':hashlib.sha256(source).hexdigest(),
                'equation':('da/dt = P_K[div(M L grad(delta F/delta U)) + M R - external_sink]'
                            if self.accessibility is not None else
                            'da/dt = P_K[div(L grad(delta F/delta U)) + R - external_sink]'),
                'autonomous':self.config.get('autonomous',True)}

    # Return cached JSON text of the RHS descriptor; this is archive data, not executable
    # generated Python.
    def source(self):
        if not hasattr(self,'_source'):
            self._source=json.dumps(self.descriptor(),ensure_ascii=False,indent=2,allow_nan=False)
        return self._source

    # Return cached basis/field/BC/provider metadata arrays for the archive writer,
    # including accessibility when present.
    def archive_extras(self):
        from engine.archive import json_array
        if hasattr(self,'_extras'): return self._extras
        self._extras={'basis_json':json_array(self.basis.descriptor()),
                'field_metadata_json':json_array(self.fields),
                'boundary_conditions_json':json_array({f['id']:f['bc'] for f in self.fields}),
                'rhs_provider_json':np.array(self.source())}
        if self.accessibility is not None: self._extras['accessibility']=self.accessibility.copy()
        return self._extras


# Restore a provider and verify its model-source hash. A mismatch requires the recorded
# source, not silently substituting today's model.
def from_descriptor(d):
    if d.get('format')!=FORMAT: raise ValueError('Unknown RHS provider format')
    provider=GalerkinProvider(d['config'],d['fields'],TensorBasis.from_descriptor(d['basis']),d.get('forcing'),d.get('accessibility'))
    # The hash covers the model file bytes, including comments. For old-run
    # Newton/RHS analysis use its recorded model source in a separate checkout;
    # do not disable this check to silently analyze a different equation.
    if d.get('model_source_sha256') != provider.descriptor()['model_source_sha256']:
        raise ValueError('Model source differs from archived segment; restore its recorded model file')
    return provider
