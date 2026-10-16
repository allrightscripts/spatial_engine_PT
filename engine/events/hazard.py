"""Random-time-change clock, independent of spatial solver and event maps.

One Exp(1) threshold is drawn from the run RNG. H'=Lambda(U,t) is integrated
alongside a'=G(a); solve_ivp locates H=E. H persists across unrelated events.
Only an actual stochastic birth renews this clock. Disabled means NO RNG draw.
"""
from copy import deepcopy
import numpy as np
from ..model.event_rules import evaluate_birth_rate


# Change the intensity in model/event_rules.py, not the clock algorithm here.
# Drawing random numbers in an RHS callback would make timing depend on solver
# retries and tolerances rather than the intended stochastic event process.
class BirthClock:
    # Initialize integrated hazard, birth count and RNG state; draw an exponential
    # threshold only if the clock is enabled.
    def __init__(self, config, rng, seed):
        self.config = deepcopy(config)
        self.rng = rng
        self.seed = seed
        self.count = 0
        self.value = 0.0
        self.threshold = None
        limit = config.get('max_births')
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('max_births must be a nonnegative integer')
        if self.enabled:
            self.renew()

    # Check the enable flag and optional max_births limit without consuming RNG or
    # changing clock state.
    @property
    def enabled(self):
        limit = self.config.get('max_births')
        return self.config.get('enabled', False) and (limit is None or self.count < limit)

    # Reset accumulated hazard and draw a positive Exp(1) threshold, recording the
    # generator state before the draw.
    def renew(self):
        self.value = 0.0
        self.draw_state = deepcopy(self.rng.bit_generator.state)
        self.threshold = float(self.rng.exponential())
        if self.threshold <= 0:
            self.threshold = float(np.nextafter(0.0, 1.0))

    def integral_clock(self, provider):
        """Generic integration hook: auxiliary scalar is never a physical DOF."""
        # Reconstruct the trial physical state and evaluate total birth intensity. Must
        # remain deterministic across adaptive solver retries.
        def rate(t, a):
            U = provider.physical_state(a)
            return evaluate_birth_rate(self.config, U, provider.fields, provider.grid, t)
        return {'rate': rate, 'initial': self.value, 'threshold': self.threshold}

    # Return the current threshold, integrated hazard, birth number and RNG provenance as
    # event metadata.
    def audit(self):
        return {'seed': self.seed, 'bit_generator': type(self.rng.bit_generator).__name__,
                'threshold': self.threshold, 'integrated_hazard': self.value,
                'threshold_rng_state_before': self.draw_state, 'birth_number': self.count,
                'rate_function': self.config.get('rate_function', 'default_p_birth_rate'),
                'rate_parameters': deepcopy(self.config.get('rate_parameters', {}))}

    # Count a committed stochastic birth and renew the threshold only if further births
    # remain enabled.
    def consumed(self):
        self.count += 1
        if self.enabled:
            self.renew()
