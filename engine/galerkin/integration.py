"""Keep adaptive solver history between control slabs of one unchanged ODE.

Only public SciPy solver classes are used. A step stops at every control
boundary: detectors never inspect interpolated states from beyond that boundary.
A new segment owns a new session, so events cannot reuse a stale Jacobian.
LSODA retains the solve_ivp path because its Fortran continuation is different.
"""
from types import SimpleNamespace
import numpy as np
from scipy.integrate import Radau, BDF, RK45, DOP853
from scipy.optimize import brentq


class IntegrationSession:
    methods = {'Radau': Radau, 'BDF': BDF, 'RK45': RK45, 'DOP853': DOP853}

    # Start without a live solver. One session belongs to one unchanged segment; never
    # reuse it after a structural event.
    def __init__(self):
        self.solver = None
        self.settings = None

    def solve(self, rhs, interval, initial, *, method, t_eval, rtol, atol,
              max_step, events=None):
        """solve_ivp-shaped result; optional upward terminal integral-clock root."""
        start, end = interval
        settings = (method, rtol, atol, max_step)
        s = self.solver
        if s is None:
            before = (0, 0, 0)
            s = self.methods[method](rhs, start, initial, end, rtol=rtol,
                                     atol=atol, max_step=max_step)
            self.solver, self.settings = s, settings
        else:
            if (self.settings != settings or s.status != 'finished' or
                    s.t != start or not np.array_equal(s.y, initial)):
                raise ValueError('Integration session must continue the same unchanged state and settings')
            before = (s.nfev, s.njev, s.nlu)
            # Retain step-size control, Jacobian and BDF history. SciPy handles
            # a step shortened by t_bound with its usual step-size machinery.
            s.t_bound = end
            s.status = 'running'
        times = [start]
        values = [np.asarray(initial).copy()]
        cursor = 1
        roots, root_values = [], []
        message = None
        while s.status == 'running':
            old_event = events(s.t, s.y) if events is not None else None
            message = s.step()
            if s.status == 'failed':
                break
            stop = s.t
            interpolation = None
            if events is not None and old_event <= 0 <= events(s.t, s.y):
                interpolation = s.dense_output()
                eps = 4*np.finfo(float).eps
                stop = brentq(lambda t: events(t, interpolation(t)), s.t_old, s.t,
                              xtol=eps, rtol=eps)
                roots.append(stop)
                root_values.append(interpolation(stop))
            # Construct dense output only for actual saved samples/root finding.
            while cursor < len(t_eval) and t_eval[cursor] <= stop:
                t = t_eval[cursor]
                if t == s.t:
                    y = s.y.copy()
                else:
                    if interpolation is None:
                        interpolation = s.dense_output()
                    y = interpolation(t)
                times.append(t)
                values.append(y)
                cursor += 1
            if roots:
                # The caller closes the segment at this root and applies its
                # event map. The trial endpoint is never carried across it.
                self.solver = None
                break
        return SimpleNamespace(
            success=s.status != 'failed', message=message,
            t=np.asarray(times), y=np.stack(values, axis=1),
            t_events=[np.asarray(roots)], y_events=[np.asarray(root_values)],
            nfev=s.nfev-before[0], njev=s.njev-before[1], nlu=s.nlu-before[2])
