# Integrate cosine Galerkin equations with an adaptive SciPy solver and save an NPZ archive.
# Recommended for stiff Onsager/Galerkin systems: Radau or BDF.
# Requires numpy + scipy. Run: spatial-engine integrate

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from engine.archive import write_archive


# Text file containing ORDER, DEFAULT_PARAMETERS, and dstate_dt[0], dstate_dt[1], ...
# Legacy du_dt[...] / u[...] files are also accepted.
# A relative path is resolved next to THIS script, independent of terminal cwd.
# If the file is elsewhere, give its full Windows path, e.g.
EQUATIONS_PATH = "equations.txt"  # historical API setting; CLI requires an explicit path

# Spatial profiles at t=0, projected onto the modes in ORDER before RK4.
# Allowed: x1, x2, x3 (as applicable), Lx1... and other parameters,
# pi, sin, cos, exp, sqrt, log and other functions listed below.
# ============================================================
# INITIAL FIELD TEMPLATES
#
# Coordinates available inside INITIAL_FIELDS:
#   1D: x1
#   2D: x1, x2
#   3D: x1, x2, x3
#
# A      = amplitude / total scale
# x0,y0,z0 = center
# sigma  = Gaussian width
# R      = sphere / disk radius
# ============================================================


# --- 1. DIRAC-LIKE DROP --------------------------------------
# Numerically use a narrow normalized Gaussian instead of a true Dirac delta.
#
# 1D:
# lambda x1: A/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x1-x0)**2/(2*sigma**2))
#
# 2D:
# lambda x1,x2: A/(2*np.pi*sigma**2) * np.exp(-((x1-x0)**2 + (x2-y0)**2)/(2*sigma**2))
#
# 3D:
# lambda x1,x2,x3: A/((2*np.pi)**1.5*sigma**3) * np.exp(-((x1-x0)**2 + (x2-y0)**2 + (x3-z0)**2)/(2*sigma**2))
#
# Here A is approximately the TOTAL MASS:
#     integral u(x) dx = A
#
# Smaller sigma -> sharper approximation to Dirac delta.
# IMPORTANT: sigma must still be resolvable by the retained Fourier modes.


# --- 2. GAUSSIAN BLOB ----------------------------------------
# Convenient smooth localized initial condition.
#
# 1D:
# lambda x1: A * np.exp(-(x1-x0)**2/(2*sigma**2))
#
# 2D:
# lambda x1,x2: A * np.exp(-((x1-x0)**2 + (x2-y0)**2)/(2*sigma**2))
#
# 3D:
# lambda x1,x2,x3: A * np.exp(-((x1-x0)**2 + (x2-y0)**2 + (x3-z0)**2)/(2*sigma**2))
#
# Here A is the PEAK HEIGHT:
#     u(center) = A


# --- 3. DISK / SPHERE ----------------------------------------
# Constant field inside radius R, zero outside.
#
# 1D interval:
# lambda x1: A * ((x1-x0)**2 <= R**2)
#
# 2D disk:
# lambda x1,x2: A * (((x1-x0)**2 + (x2-y0)**2) <= R**2)
#
# 3D sphere:
# lambda x1,x2,x3: A * (((x1-x0)**2 + (x2-y0)**2 + (x3-z0)**2) <= R**2)
#
# Here A is the field value INSIDE the object.
#
# NOTE:
# A sharp sphere/disk contains arbitrarily high spatial frequencies.
# A finite cosine expansion will therefore show Gibbs oscillations
# near the boundary. This is expected, not necessarily a numerical bug.


# --- EXAMPLE: 2D Gaussian blob -------------------------------
#
A = 1.0
x0 = 0.5
y0 = 0.5
sigma = 0.15

INITIAL_FIELDS = {
    "u": lambda x1, x2: A * np.exp(
        -((x1-x0)**2 + (x2-y0)**2)/(2*sigma**2)
    ),
    "v": lambda x1, x2: 0.0*x1,
}

# --- EXAMPLE: 2D Dirac-like drop -----------------------------
#
# A = 1.0
# x0 = 0.5
# y0 = 0.5
# sigma = 0.02
#
# INITIAL_FIELDS = {
#     "u": lambda x1,x2: A/(2*np.pi*sigma**2) * np.exp(
#         -((x1-x0)**2 + (x2-y0)**2)/(2*sigma**2)
#     ),
#     "v": lambda x1,x2: 0.0*x1,
# }


# --- EXAMPLE: 2D disk ----------------------------------------
#
# A = 1.0
# x0 = 0.5
# y0 = 0.5
# R = 0.15
#
# INITIAL_FIELDS = {
#     "u": lambda x1,x2: A*(
#         ((x1-x0)**2 + (x2-y0)**2) <= R**2
#     ),
#     "v": lambda x1,x2: 0.0*x1,
# }
# ============================================================

# Optional corrections AFTER projection: (field, mode tuple) -> coefficient.
# An override replaces the projected coefficient for that one mode.
INITIAL_COEFFICIENTS = {}
QUADRATURE_POINTS = 64  # Gauss–Legendre points per spatial axis

PARAMETER_OVERRIDES = {}  # e.g. {"Du": 0.15, "Lx1": 2.0}
T_START = 0.0
T_END = 5.0

# Adaptive integrator settings.
# Radau/BDF are appropriate when high Fourier modes make the Galerkin ODE stiff.
METHOD = "Radau"          # "Radau", "BDF", "RK45", "DOP853", "LSODA"
RTOL = 1e-7
ATOL = 1e-9
MAX_STEP = 0.01           # upper bound only; solver may take MUCH smaller steps
SAVE_DT = 0.1             # spacing of snapshots written to trajectory.npz
FIRST_STEP = None         # e.g. 1e-5, or None for automatic choice

OUTPUT_ARCHIVE = "trajectory.npz"  # historical setting; CLI requires --output


_NUMPY_NAMES = {"pi", "e", "sin", "cos", "tan", "exp", "log", "sqrt",
                "asin", "acos", "atan", "arcsin", "arccos", "arctan",
                "sinh", "cosh", "tanh",
                "floor", "ceil"}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
              ast.UAdd, ast.USub)


# Inspect the expression AST against the allowed syntax before compilation; this is the
# restricted legacy equation interface, not arbitrary Python execution.
def _check_expression(tree: ast.AST, allowed_names: set[str], size: int) -> None:
    # Check the generated equations before evaluating their right-hand sides.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp,
                             ast.Call, ast.Name, ast.Subscript, ast.Attribute,
                             ast.Constant) + _OPERATORS):
            pass
        else:
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown name in equation: {node.id}")
        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id in ("state", "u")
                    and isinstance(node.slice, ast.Constant)
                    and type(node.slice.value) is int and 0 <= node.slice.value < size):
                raise ValueError("State indices must have the form state[0], state[1], ... (legacy u[...] is also accepted)")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"
                    and node.attr in _NUMPY_NAMES):
                raise ValueError("Only supported np mathematical functions are allowed")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Attribute)
                    and node.func.attr in _NUMPY_NAMES - {"pi", "e"}
                    and not node.keywords):
                raise ValueError("Only direct np mathematical function calls are allowed")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError("Only real numeric constants are allowed in equations")


# Read ORDER/default parameters and RHS expressions, or detect a provider descriptor.
# Validate and prepare the representation used by the numerical caller.
def load_equations(source: str):
    statements = ast.parse(source).body
    metadata = {}
    rhs = {}
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Input must contain only ORDER, DEFAULT_PARAMETERS and dstate_dt[i] assignments")
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in ("ORDER", "DEFAULT_PARAMETERS"):
            if target.id in metadata:
                raise ValueError(f"Repeated {target.id}")
            metadata[target.id] = ast.literal_eval(statement.value)
        elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
              and target.value.id in ("dstate_dt", "du_dt") and isinstance(target.slice, ast.Constant)
              and type(target.slice.value) is int):
            index = target.slice.value
            if index in rhs:
                raise ValueError(f"Repeated derivative assignment at index {index}")
            rhs[index] = statement.value
        else:
            raise ValueError("Unexpected assignment in generated text")

    order = metadata.get("ORDER")
    defaults = metadata.get("DEFAULT_PARAMETERS")
    if not isinstance(order, list) or not order or not isinstance(defaults, dict):
        raise ValueError("Missing or invalid ORDER / DEFAULT_PARAMETERS")
    if any(not isinstance(item, tuple) or len(item) != 2
           or not isinstance(item[0], str) or not isinstance(item[1], tuple)
           or any(type(k) is not int or k < 0 for k in item[1]) for item in order):
        raise ValueError("ORDER must contain (field, mode tuple) pairs")
    if len(set(order)) != len(order) or len({len(index) for _, index in order}) != 1:
        raise ValueError("ORDER must contain unique, equally dimensioned modes")
    if set(rhs) != set(range(len(order))):
        raise ValueError("Expected exactly one dstate_dt[i] for every item in ORDER")
    if any(not isinstance(k, str) or not k.isidentifier()
           or type(v) not in (float, int) for k, v in defaults.items()):
        raise ValueError("Parameters must be numeric with Python identifier names")

    allowed = {"state", "u", "t", "np"} | set(defaults)
    compiled = []
    for i in range(len(order)):
        expression = ast.Expression(rhs[i])
        _check_expression(expression, allowed, len(order))
        compiled.append(compile(ast.fix_missing_locations(expression), f"dstate_dt[{i}]", "eval"))
    return order, defaults, compiled


_PROFILE_FUNCTIONS = {name: getattr(np, name) for name in
                      _NUMPY_NAMES - {"pi", "e"} if hasattr(np, name)}


# Compile an allowed initial-profile text expression; coordinates and parameters
# are supplied during projection. Callable profiles are handled by the caller.
def _compile_profile(expression: str, parameters: dict, dimension: int):
    # Compile a spatial initial profile with only numeric NumPy operations.
    tree = ast.parse(expression, mode="eval")
    coordinates = {f"x{i}" for i in range(1, dimension + 1)}
    allowed = coordinates | set(parameters) | set(_PROFILE_FUNCTIONS) | {"pi", "e"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp,
                                 ast.Call, ast.Name, ast.Constant) + _OPERATORS):
            raise ValueError(f"Unsupported spatial profile syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ValueError(f"Unknown name in initial profile: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in _PROFILE_FUNCTIONS
                                               and not node.keywords):
            raise ValueError("Use supported functions such as cos(...), sin(...), exp(...)")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError("Initial profiles must contain real numeric constants")
    return compile(tree, "initial profile", "eval")


# Numerically project configured physical initial fields into the ORDER cosine modes. This
# initializes coefficients, not a simulation step.
def project_initial_fields(profiles: dict, order: list, parameters: dict,
                           quadrature_points: int) -> np.ndarray:
    # Project spatial initial profiles onto the cosine Galerkin basis.
    # Profiles may be:
    #   1) callables: lambda x1, x2: ...
    #   2) legacy strings: "exp(-((x1-0.5)**2 + ...))"
    # Missing physical fields are interpreted as identically zero.
    dimension = len(order[0][1])
    fields = {field for field, _ in order}
    if profiles is None:
        return np.zeros(len(order), dtype=float)
    if not isinstance(profiles, dict):
        raise ValueError("INITIAL_FIELDS must be a dict, for example {'u': lambda x1,x2: ...}")
    unknown = set(profiles) - fields
    if unknown:
        raise ValueError(f"Unknown fields in INITIAL_FIELDS: {sorted(unknown)}; expected {sorted(fields)}")
    if type(quadrature_points) is not int or quadrature_points < 2:
        raise ValueError("QUADRATURE_POINTS must be an integer of at least 2")

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    mode_values = [sorted({idx[axis] for _, idx in order}) for axis in range(dimension)]
    lengths = [parameters[f"Lx{i}"] for i in range(1, dimension + 1)]
    coordinates = [(nodes + 1) * length / 2 for length in lengths]
    mesh = np.meshgrid(*coordinates, indexing="ij", sparse=True)
    grid_shape = (quadrature_points,) * dimension

    # The weights already include 1/||phi_k||^2 and [−1,1] -> [0,L].
    basis = [(weights[:, None] / 2) * np.cos(
                 np.pi * (nodes[:, None] + 1) / 2 * np.array(modes)[None, :])
             * np.array([1 if k == 0 else 2 for k in modes])
             for modes in mode_values]

    spatial_labels = "abc"[:dimension]
    modal_labels = "ijk"[:dimension]
    expression = (spatial_labels + "," + ",".join(a + b for a, b in
                  zip(spatial_labels, modal_labels)) + "->" + modal_labels)
    mode_positions = [{mode: pos for pos, mode in enumerate(modes)} for modes in mode_values]

    values_by_field = {}
    coordinate_args = tuple(mesh[i] for i in range(dimension))

    for field in fields:
        # Omitted fields are zero by definition.
        if field not in profiles or profiles[field] is None:
            values_by_field[field] = np.zeros(tuple(len(m) for m in mode_values), dtype=float)
            continue

        profile = profiles[field]
        if callable(profile):
            try:
                raw = profile(*coordinate_args)
            except Exception as exc:
                raise ValueError(f"Could not evaluate callable INITIAL_FIELDS[{field!r}]") from exc
        elif isinstance(profile, str):
            compiled_profile = _compile_profile(profile, parameters, dimension)
            namespace = {"pi": np.pi, "e": np.e, **_PROFILE_FUNCTIONS,
                         **parameters, **{f"x{i}": mesh[i - 1]
                                        for i in range(1, dimension + 1)}}
            raw = eval(compiled_profile, {"__builtins__": {}}, namespace)
        elif isinstance(profile, (int, float, np.number)):
            raw = float(profile)
        else:
            raise ValueError(
                f"INITIAL_FIELDS[{field!r}] must be a lambda/callable, string, or number; "
                f"got {type(profile).__name__}"
            )

        try:
            samples = np.broadcast_to(np.asarray(raw, dtype=float), grid_shape)
        except ValueError as exc:
            raise ValueError(
                f"INITIAL_FIELDS[{field!r}] returned shape {np.shape(raw)}, "
                f"which cannot be broadcast to spatial grid {grid_shape}"
            ) from exc
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"Non-finite values in initial profile for {field}")
        values_by_field[field] = np.einsum(expression, samples, *basis, optimize=True)

    return np.array([values_by_field[field][tuple(mode_positions[axis][k]
                      for axis, k in enumerate(idx))]
                     for field, idx in order], dtype=float)


# Integrate legacy generated equations with adaptive solve_ivp and assemble a standard
# archive dictionary; caller decides when to write it.
def integrate(source: str, initial: dict, overrides: dict, t_start: float,
              t_end: float, method: str = "Radau", rtol: float = 1e-7,
              atol: float = 1e-9, max_step: float = np.inf,
              save_dt: float = 0.1, first_step: float | None = None,
              initial_fields: dict | None = None, quadrature_points: int = 64):
    order, defaults, compiled = load_equations(source)
    parameters = {**defaults, **overrides}
    if set(overrides) - set(defaults):
        raise ValueError(f"Unknown parameter overrides: {set(overrides) - set(defaults)}")
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for v in parameters.values()):
        raise ValueError("All parameters must be finite real numbers")

    dimension = len(order[0][1])
    if any(parameters.get(f"Lx{i}", 0) <= 0 for i in range(1, dimension + 1)):
        raise ValueError("Each spatial length Lx1, Lx2, ... must be positive")
    if set(initial) - set(order):
        raise ValueError(f"Initial coefficients absent from ORDER: {set(initial) - set(order)}")
    if any(not math.isfinite(value) for value in initial.values()):
        raise ValueError("Initial coefficients must be finite")

    allowed_methods = {"Radau", "BDF", "RK45", "DOP853", "LSODA"}
    if method not in allowed_methods:
        raise ValueError(f"METHOD must be one of {sorted(allowed_methods)}")
    if not (math.isfinite(t_start) and math.isfinite(t_end) and t_end > t_start):
        raise ValueError("Need finite T_START < T_END")
    if not (math.isfinite(rtol) and rtol > 0 and math.isfinite(atol) and atol > 0):
        raise ValueError("RTOL and ATOL must be positive finite numbers")
    if not ((math.isfinite(max_step) and max_step > 0) or max_step == np.inf):
        raise ValueError("MAX_STEP must be positive")
    if not (math.isfinite(save_dt) and save_dt > 0):
        raise ValueError("SAVE_DT must be positive")
    if first_step is not None and not (math.isfinite(first_step) and first_step > 0):
        raise ValueError("FIRST_STEP must be None or positive")

    state0 = (project_initial_fields(initial_fields, order, parameters, quadrature_points)
              if initial_fields is not None else np.zeros(len(order), dtype=float))
    for item, coefficient in initial.items():
        state0[order.index(item)] = coefficient
    initial_state = state0.copy()

    safe_globals = {"__builtins__": {}}
    diagnostic = {"t": float(t_start), "max_state": float(np.max(np.abs(state0))),
                  "max_derivative": None, "largest_state_index": int(np.argmax(np.abs(state0)))}

    # Evaluate compiled legacy equations for the current state/time and reject nonfinite
    # derivatives before the solver proceeds.
    def derivative(t, state):
        diagnostic["t"] = float(t)
        diagnostic["max_state"] = float(np.max(np.abs(state)))
        diagnostic["largest_state_index"] = int(np.argmax(np.abs(state)))

        # Generated polynomial expressions can overflow during a rejected trial step.
        # Suppress NumPy's repetitive warnings; non-finite values are detected below.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            variables = {"np": np, "t": t, "state": state, "u": state, **parameters}
            result = np.array([eval(expr, safe_globals, variables) for expr in compiled],
                              dtype=float)

        finite = np.isfinite(result)
        if not np.all(finite):
            bad = np.flatnonzero(~finite)
            idx = diagnostic["largest_state_index"]
            field, mode = order[idx]
            raise FloatingPointError(
                "\nNon-finite RHS encountered.\n"
                f"  t                = {t:.16g}\n"
                f"  max |state|      = {diagnostic['max_state']:.6e}\n"
                f"  largest state    = state[{idx}] = {field}_{'_'.join(map(str, mode))}\n"
                f"  bad RHS indices  = {bad[:12].tolist()}"
                + (" ..." if len(bad) > 12 else "")
                + "\nThis can mean the solution itself is diverging, or that even the "
                  "adaptive solver reached values outside float64 range."
            )
        diagnostic["max_derivative"] = float(np.max(np.abs(result)))
        return result

    # Save on a regular physical-time grid; adaptive internal steps are independent of this.
    n_save = int(math.floor((t_end - t_start) / save_dt))
    t_eval = t_start + np.arange(n_save + 1, dtype=float) * save_dt
    if t_eval.size == 0 or t_eval[-1] < t_end - 1e-14 * max(1.0, abs(t_end)):
        t_eval = np.append(t_eval, t_end)
    else:
        t_eval[-1] = t_end

    kwargs = dict(method=method, rtol=rtol, atol=atol, max_step=max_step, t_eval=t_eval)
    if first_step is not None:
        kwargs["first_step"] = first_step

    try:
        solution = solve_ivp(derivative, (t_start, t_end), state0, **kwargs)
    except FloatingPointError:
        raise
    except Exception as exc:
        idx = diagnostic["largest_state_index"]
        field, mode = order[idx]
        raise RuntimeError(
            f"Integrator failed near t={diagnostic['t']:.16g}; "
            f"max|state|={diagnostic['max_state']:.6e}; "
            f"largest={field}_{'_'.join(map(str, mode))}; "
            f"max|dstate|={diagnostic['max_derivative']}"
        ) from exc

    if not solution.success:
        idx = diagnostic["largest_state_index"]
        field, mode = order[idx]
        raise RuntimeError(
            "\nAdaptive integration failed.\n"
            f"  method           = {method}\n"
            f"  last solver t    = {solution.t[-1] if solution.t.size else t_start:.16g}\n"
            f"  diagnostic t     = {diagnostic['t']:.16g}\n"
            f"  max |state|      = {diagnostic['max_state']:.6e}\n"
            f"  largest state    = {field}_{'_'.join(map(str, mode))}\n"
            f"  max |dstate|     = {diagnostic['max_derivative']}\n"
            f"  solver message   = {solution.message}"
        )

    states = solution.y.T
    if not np.all(np.isfinite(states)):
        raise FloatingPointError("Solver returned non-finite states")

    info = {
        "method": method,
        "adaptive": True,
        "dimension": dimension,
        "time_start": t_start,
        "time_end": t_end,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "first_step": first_step,
        "save_dt": save_dt,
        "saved_snapshots": int(solution.t.size),
        "nfev": int(solution.nfev),
        "njev": None if solution.njev is None else int(solution.njev),
        "nlu": None if solution.nlu is None else int(solution.nlu),
        "domain": {f"x{i}": [0, parameters[f"Lx{i}"]]
                   for i in range(1, dimension + 1)},
        "initial_projection": ("Gauss-Legendre" if initial_fields is not None else "modal"),
        "quadrature_points_per_axis": quadrature_points if initial_fields is not None else None,
    }
    return {
        "time": solution.t,
        "coefficients": states,
        "initial_coefficients": initial_state,
        "initial_fields_json": np.array(json.dumps({"note": "Initial callable fields were projected to initial_coefficients"})),
        "fields": np.array([field for field, _ in order], dtype=str),
        "modes": np.array([index for _, index in order], dtype=np.int64),
        "order_json": np.array(json.dumps(order)),
        "parameters_json": np.array(json.dumps(parameters)),
        "metadata_json": np.array(json.dumps(info)),
        "equations_text": np.array(source),
    }


def integrate_provider(provider, initial, t_start, t_end, method="Radau",
                       rtol=1e-7, atol=1e-9, max_step=np.inf, save_dt=.1,
                       output_times=None, integral_clock=None, integration_session=None):
    """Additional universal callable-RHS path; legacy integrate() stays unchanged.

    Randomness and structural changes must be handled by the caller BETWEEN
    integrations. A provider's order and RHS are immutable throughout this call.
    Returned keys match the scalar-equation path. By default use solve_ivp;
    an optional segment-owned session retains adaptive history across slabs.
    Never share a session between different providers or event maps.
    """
    from engine.archive import json_array
    initial=np.asarray(initial,float)
    if initial.shape!=(len(provider.order),) or not np.isfinite(initial).all():
        raise ValueError("Invalid provider initial state")
    if method not in {"RK45","DOP853","Radau","BDF","LSODA"}:
        raise ValueError("Unsupported solve_ivp method")
    if not (np.isfinite(t_start) and np.isfinite(t_end) and t_end>t_start):
        raise ValueError("Need finite t_start < t_end")
    if not (np.isfinite(save_dt) and save_dt>0 and rtol>0 and atol>0 and max_step>0):
        raise ValueError("Invalid integrator settings")
    if output_times is None:
        count=int(np.floor((t_end-t_start)/save_dt))
        times=t_start+np.arange(count+1)*save_dt
        times=times[times<t_end-1e-13]
        times=np.r_[times,t_end]
    else:
        times=np.asarray(output_times,float)
        if times.ndim!=1 or len(times)<2 or times[0]!=t_start or times[-1]!=t_end or np.any(np.diff(times)<=0):
            raise ValueError("Output times must increase from segment start to end")
    # Optional generic nonnegative integrated intensity. It is an auxiliary
    # solver variable, NEVER a coefficient/ORDER entry or a provider equation.
    # No random number is drawn here: the caller supplies an Exp(1) threshold.
    rhs = provider.rhs
    y0 = initial
    events = None
    if integral_clock is not None:
        clock = integral_clock
        h0, threshold = float(clock['initial']), float(clock['threshold'])
        if not (np.isfinite(h0) and np.isfinite(threshold) and 0 <= h0 < threshold):
            raise ValueError('Integral clock needs 0 <= initial < finite threshold')
        # Evaluate the provider's coefficient RHS and, when enabled, append the auxiliary
        # integrated-hazard derivative. The clock coordinate is not a physical field.
        def rhs(t, y):
            derivative = provider.rhs(t, y[:-1])
            rate = float(clock['rate'](t, y[:-1]))
            if not np.isfinite(rate) or rate < 0:
                raise ValueError('Integral intensity must be finite and nonnegative')
            return np.r_[derivative, rate]
        # Return accumulated hazard minus its threshold for upward terminal root
        # detection; do not draw randomness during root finding.
        def stop(t, y):
            return y[-1] - threshold
        stop.terminal = True
        stop.direction = 1
        events = stop
        y0 = np.r_[initial, h0]
    solve = solve_ivp
    if integration_session is not None and method in integration_session.methods:
        solve = integration_session.solve
    solution=solve(rhs,(t_start,t_end),y0,method=method,
                       t_eval=times,rtol=rtol,atol=atol,max_step=max_step,
                       events=events)
    if not solution.success: raise RuntimeError(solution.message)
    if not np.isfinite(solution.y).all(): raise FloatingPointError("Nonfinite provider trajectory")
    clock_result = None
    if integral_clock is not None:
        reached = bool(len(solution.t_events[0]))
        if reached:
            root = float(solution.t_events[0][0])
            endpoint = solution.y_events[0][0]
            # A terminal root need not be one of the requested saved times.
            if solution.t[-1] != root:
                solution.t = np.r_[solution.t, root]
                solution.y = np.column_stack((solution.y, endpoint))
            else:
                solution.y[:, -1] = endpoint
        clock_result = {'value': float(solution.y[-1,-1]), 'reached': reached,
                        'threshold': threshold}
        solution.y = solution.y[:-1]
    descriptor=provider.archive_extras()
    representation=getattr(provider,'representation','cosine_galerkin')
    meta={'method':method,'rtol':rtol,'atol':atol,'max_step':None if max_step==np.inf else max_step,
          'representation':representation,'dimension':len(provider.order[0][1]),
          'rhs_backend':'provider','nfev':int(solution.nfev),'njev':int(solution.njev),'nlu':int(solution.nlu)}
    meta['solver_continuation'] = solve != solve_ivp
    if clock_result is not None: meta['integral_clock'] = clock_result
    return {'time':solution.t,'coefficients':solution.y.T,'initial_coefficients':initial.copy(),
            'fields':np.array([f for f,_ in provider.order]),'modes':np.array([k for _,k in provider.order],dtype=np.int64),
            'order_json':json_array(provider.order),'parameters_json':json_array(provider.parameters),
            'metadata_json':json_array(meta),'equations_text':np.array(provider.source()),**descriptor}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive integration of Galerkin equations")
    parser.add_argument("--equations", type=Path, required=True,
                        help="Input legacy equations text or RHS provider descriptor")
    parser.add_argument("--initial-archive", type=Path, help="Provider initial state and interval")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path")
    args = parser.parse_args()
    source_path = Path(args.equations or EQUATIONS_PATH).expanduser()
    if not source_path.is_absolute():
        source_path = source_path.resolve()
    if not source_path.is_file():
        parser.error(f"Equations file not found: {source_path}. Pass an existing --equations path")
    source = source_path.read_text(encoding="utf-8-sig")
    # Provider CLI uses its numeric initial state from --initial-archive.
    from engine.numerics.rhs import is_provider_source, load_provider
    if is_provider_source(source):
        if args.initial_archive is None:
            parser.error("Provider mode requires --initial-archive with initial_coefficients")
        from engine.archive import read_archive
        previous = read_archive(args.initial_archive)
        provider = load_provider(source)
        archive = integrate_provider(provider, previous['initial_coefficients'],
                                     float(previous['time'][0]), float(previous['time'][-1]),
                                     method=METHOD, rtol=RTOL, atol=ATOL, max_step=MAX_STEP, save_dt=SAVE_DT)
    else:
        archive = integrate(source, INITIAL_COEFFICIENTS, PARAMETER_OVERRIDES,
                        T_START, T_END, METHOD, RTOL, ATOL, MAX_STEP, SAVE_DT, FIRST_STEP,
                        initial_fields=INITIAL_FIELDS, quadrature_points=QUADRATURE_POINTS)
    field_names = sorted(set(archive["fields"].tolist()))
    parameters = json.loads(archive["parameters_json"].item())
    print(f"Equations: {source_path}")
    print(f"Fields: {field_names}; modes: {len(archive['fields'])}; parameters: {parameters}")
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = output_path.resolve()
    write_archive(output_path, archive)
    print(f"Saved {output_path}: {archive['coefficients'].shape[0]} snapshots, "
          f"{archive['coefficients'].shape[1]} coefficients, "
          f"t={archive['time'][0]:g}..{archive['time'][-1]:g}")
