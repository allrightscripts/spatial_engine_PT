"""Cheap snapshot diagnostics. Component counts reuse the detector's cache."""
import numpy as np

# Compute masses, family counts, entropy and functional value from current state.
# Component counts reuse the detector cache and include their last-check times.
def observe(sim,t):
    mass=sim.u.reshape(len(sim.registry),-1).sum(axis=1)*sim.grid.volume
    total=float(mass.sum()); fractions=mass[mass>0]/total if total>0 else np.array([])
    energy=sim.model.functional.value(sim.u,sim.registry,sim.grid,t)
    return {'mass':mass.copy(),'total_mass':total,
            'active_P':sum(f['active'] and f['family']=='P' for f in sim.registry),
            'active_T':sum(f['active'] and f['family']=='T' for f in sim.registry),
            'entropy':float(-np.sum(fractions*np.log(fractions))),
            'functional':float(energy) if energy is not None else 0.,
            'functional_available':int(energy is not None),
            'components':np.array([sim.detector.last_counts.get(f['id'],-1) if f['active'] else 0 for f in sim.registry]),
            'components_checked_at':np.array([sim.detector.last_time.get(f['id'],-1) for f in sim.registry]),
            'active':np.array([f['active'] for f in sim.registry],dtype=np.int8)}
