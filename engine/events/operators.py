"""Spatial geometry and replaceable event handlers, separate from integration."""
from copy import deepcopy
import numpy as np
from .base import NucleationEvent, MutationEvent


def spatial_mask(grid, geometry, t=0.0):
    """Backward-compatible entry point into the shared Python mask registry."""
    from ..model.masks import evaluate_mask
    return evaluate_mask(grid, geometry, t)


def suppress(sim,event):
    """Impulse suppression, optionally held until time+duration.

    A held mask multiplies by (1-strength*mask) after each integration substep;
    partial held suppression is step-dependent, intentionally not a rate law.
    Use duration=0 for a time-step independent impulse.
    """
    if not 0<=event.strength<=1: raise ValueError('Suppression strength must be [0,1]')
    mask=spatial_mask(sim.grid,event.geometry,event.time)
    ids=sim.target_indices(event.targets)
    sim.u[ids] *= 1-event.strength*mask
    duration=float(event.parameters.get('duration',0))
    if duration<0 or not np.isfinite(duration): raise ValueError('Invalid duration')
    if duration:
        sim.held.append({'event':event,'end':event.time+duration,'mask':mask})


# Create a field from amplitude times a geometric mask using the existing nucleation
# event. Set event.targets to the newborn ID; distinct from per_T P birth.
def nucleate(sim,event):
    p=event.parameters
    name=p.get('id',sim.next_name(p.get('family','P')))
    amplitude=event.strength
    if amplitude<0: raise ValueError('Negative nucleation amplitude')
    sim.add_field(name,p.get('family','P'),amplitude*spatial_mask(sim.grid,event.geometry,event.time),event.time,
                  p.get('field_parameters',{}),parent=p.get('parent_id'),bc=p.get('bc'))
    event.targets=[name]


def mutate(sim,event):
    """Explicit parameter replacement; stochastic perturbation happens upstream."""
    for i in sim.target_indices(event.targets):
        updated={**sim.registry[i]['parameters'],**event.parameters['updates']}
        sim.check_parameters(updated)
        sim.registry[i]['parameters']=updated


# Retire each active resolved target at event time, preserving its registry row for
# historical analysis.
def extinct(sim,event):
    for i in sim.target_indices(event.targets):
        sim.retire(i,event.time)


def manual(sim,event):
    """Minimal intervention: add a nonnegative spatial field increment."""
    if event.parameters.get('action','add') != 'add': raise ValueError('Manual action must be add')
    for i in sim.target_indices(event.targets):
        sim.u[i] += event.strength*spatial_mask(sim.grid,event.geometry,event.time)

HANDLERS={'catastrophe':suppress,'sabotage':suppress,'nucleation':nucleate,
          'mutation':mutate,'extinction':extinct,'manual':manual}

class PoissonNucleation:
    """Poisson number of births per dt; all randomness comes from sim.rng."""
    # Store configuration for the legacy per-step Poisson nucleation mechanism; it is
    # independent of the integrated P-birth clock.
    def __init__(self,config): self.config=config
    # Draw a Poisson count over dt, then random centres, using sim.rng only; apply and
    # audit each generated nucleation event.
    def step(self,sim,dt,t):
        c=self.config; rate=float(c.get('rate',0))
        if rate<0 or not np.isfinite(rate): raise ValueError('Invalid nucleation rate')
        if not rate: return
        before=deepcopy(sim.rng.bit_generator.state)
        count=int(sim.rng.poisson(rate*dt))
        for draw in range(count):
            center=[float(sim.rng.uniform(0,L)) for L in sim.grid.lengths]
            event=NucleationEvent(t,source='poisson',geometry={'type':'gaussian','center':center,'radius':c.get('radius',.05)},
                strength=c.get('amplitude',.1),parameters={'family':c.get('family','P'),'field_parameters':c.get('parameters',{}), 'bc':c.get('bc',sim.default_bc)},
                rng={'state_before_step':before,'dt':dt,'rate':rate,'count':count,'draw':draw})
            sim.apply(event)
