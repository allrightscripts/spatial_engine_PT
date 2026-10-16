"""Find stationary roots of selected Galerkin ODE equations by multi-start Newton.

Input format is the equations.txt produced by PDE_to_ODE.py:
    ORDER = [...]
    DEFAULT_PARAMETERS = {...}
    du_dt[0] = ...
    ...

Edit the CONFIG section, then run:
    spatial-engine newton
or override the equation file:
    spatial-engine newton --equations "equations.txt"

Interpretation: for selected indices I, solve du_dt[I] = 0 with respect to
selected state variables u[J]. Usually EQUATION_INDICES == VARIABLE_INDICES.
All non-selected coefficients are frozen at BACKGROUND_STATE (zero by default).
The grid supplies Newton STARTING POINTS; Newton itself is continuous.
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

# ============================== CONFIG ======================================
EQUATIONS_PATH = r"output/equations.txt"

# Numbers from the generated list, e.g. 0 -> u_0_0, 16 -> v_0_0.
EQUATION_INDICES = [0, 1, 31]
# Unknown coefficients changed by Newton. For a square system, same count as equations.
VARIABLE_INDICES = [0, 1, 31]

# Default grid for every selected variable: (min, max, number_of_start_points).
DEFAULT_GRID = (-10.0, 10.0, 100)
# Optional per-variable overrides. Key is u[index].
# Example: {0: (-1.0, 3.0, 17), 16: (0.0, 2.0, 11)}
GRID_OVERRIDES = {}

# Frozen values for coefficients NOT in VARIABLE_INDICES. Unlisted = 0.
# This also lets you freeze selected-looking modes at nonzero values if they are not unknowns.
BACKGROUND_STATE = {}  # e.g. {1: 0.2, 17: -0.1}

# Override DEFAULT_PARAMETERS from equations.txt.
PARAMETER_OVERRIDES = {}  # e.g. {"Du": 0.15, "Lx1": 2.0}
T_VALUE = 0.0             # only matters if generated RHS explicitly contains t

# Newton controls.
MAX_ITER = 80
RESIDUAL_TOL = 1e-10
STEP_TOL = 1e-12
JACOBIAN_REL_STEP = 1e-6
MIN_DAMPING = 2.0 ** -16  # backtracking line search lower bound
ROOT_MERGE_TOL = 1e-7     # roots closer than this are treated as one root

MAX_GRID_POINTS = 2_000_000  # safety against accidental Cartesian explosion
OUTPUT_CSV = "newton_roots.csv"
OUTPUT_NPZ = "newton_roots.npz"
# ===========================================================================

_NUMPY_NAMES = {"pi", "e", "sin", "cos", "tan", "exp", "log", "sqrt",
                "asin", "acos", "atan", "arcsin", "arccos", "arctan",
                "sinh", "cosh", "tanh", "floor", "ceil"}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
              ast.UAdd, ast.USub)


# Inspect the expression AST against the allowed syntax before compilation; this is the
# restricted legacy equation interface, not arbitrary Python execution.
def _check_expression(tree: ast.AST, allowed_names: set[str], size: int) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp,
                                 ast.Call, ast.Name, ast.Subscript, ast.Attribute,
                                 ast.Constant) + _OPERATORS):
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown name in equation: {node.id}")
        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "u"
                    and isinstance(node.slice, ast.Constant)
                    and type(node.slice.value) is int and 0 <= node.slice.value < size):
                raise ValueError("State indices must have form u[0], u[1], ...")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"
                    and node.attr in _NUMPY_NAMES):
                raise ValueError("Only supported np mathematical functions are allowed")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Attribute)
                    and node.func.attr in _NUMPY_NAMES - {"pi", "e"}
                    and not node.keywords):
                raise ValueError("Only direct supported np function calls are allowed")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError("Only real numeric constants are allowed")


# Read ORDER/default parameters and RHS expressions, or detect a provider descriptor.
# Validate and prepare the representation used by the numerical caller.
def load_equations(source: str):
    from engine.numerics.rhs import is_provider_source, load_provider
    if is_provider_source(source):
        provider = load_provider(source)
        return provider.order, provider.parameters, provider
    # Accept the current generator format (state/dstate_dt) while preserving
    # compatibility with older equations.txt files (u/du_dt).  Internally the
    # analysis tools use the old neutral parser names, so the mathematics is unchanged.
    source = source.replace("dstate_dt[", "du_dt[").replace("state[", "u[")
    statements = ast.parse(source).body
    metadata, rhs = {}, {}
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Input must contain only ORDER, DEFAULT_PARAMETERS and du_dt[i] assignments")
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in ("ORDER", "DEFAULT_PARAMETERS"):
            metadata[target.id] = ast.literal_eval(statement.value)
        elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
              and target.value.id == "du_dt" and isinstance(target.slice, ast.Constant)
              and type(target.slice.value) is int):
            rhs[target.slice.value] = statement.value
        else:
            raise ValueError("Unexpected assignment in generated text")

    order = metadata.get("ORDER")
    defaults = metadata.get("DEFAULT_PARAMETERS")
    if not isinstance(order, list) or not order or not isinstance(defaults, dict):
        raise ValueError("Missing/invalid ORDER or DEFAULT_PARAMETERS")
    if set(rhs) != set(range(len(order))):
        raise ValueError("Expected exactly one du_dt[i] for every ORDER item")
    allowed = {"u", "t", "np"} | set(defaults)
    compiled = []
    for i in range(len(order)):
        expr = ast.Expression(rhs[i])
        _check_expression(expr, allowed, len(order))
        compiled.append(compile(ast.fix_missing_locations(expr), f"du_dt[{i}]", "eval"))
    return order, defaults, compiled


# Format a stored field/mode entry into a readable coefficient label for reports; do not
# use the label as field identity.
def label(order, i: int) -> str:
    field, mode = order[i]
    return f"{field}_{'_'.join(map(str, mode))} [u[{i}]]"


# Build a callable subsystem using selected equations/variables and a frozen background
# state. Module settings determine which dimensions are analyzed.
def make_problem(order, defaults, compiled):
    n = len(order)
    eq = list(EQUATION_INDICES)
    var = list(VARIABLE_INDICES)
    if not eq or not var:
        raise ValueError("Select at least one equation and variable")
    if len(eq) != len(var):
        raise ValueError("Classical Newton needs a square system: same number of equations and variables")
    if len(set(eq)) != len(eq) or len(set(var)) != len(var):
        raise ValueError("Duplicate equation/variable index")
    if any(type(i) is not int or not 0 <= i < n for i in eq + var):
        raise ValueError(f"Indices must be integers in 0..{n-1}")
    if set(BACKGROUND_STATE) & set(var):
        raise ValueError("A VARIABLE_INDICES entry cannot also be frozen in BACKGROUND_STATE")
    if any(type(i) is not int or not 0 <= i < n for i in BACKGROUND_STATE):
        raise ValueError("Invalid BACKGROUND_STATE index")

    params = {**defaults, **PARAMETER_OVERRIDES}
    unknown_params = set(PARAMETER_OVERRIDES) - set(defaults)
    if unknown_params:
        raise ValueError(f"Unknown parameter overrides: {sorted(unknown_params)}")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in params.values()):
        raise ValueError("Parameters must be finite numbers")

    base = np.zeros(n, dtype=float)
    for i, value in BACKGROUND_STATE.items():
        base[i] = float(value)
    safe_globals = {"__builtins__": {}}

    # Insert trial coordinates into a copy of the frozen background, evaluate selected RHS
    # components and return NaNs for invalid trial evaluations.
    def F(x):
        state = base.copy()
        state[var] = np.asarray(x, dtype=float)
        ns = {"np": np, "t": float(T_VALUE), "u": state, **params}
        try:
            from engine.numerics.rhs import evaluate_rhs
            y = evaluate_rhs(compiled, float(T_VALUE), state, params)[eq]
        except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
            return np.full(len(eq), np.nan)
        return y

    return eq, var, params, F


# Approximate derivatives of F by centred finite differences with a scale-aware coordinate
# step; this is a local numerical Jacobian, not symbolic differentiation.
def numerical_jacobian(F, x):
    x = np.asarray(x, dtype=float)
    m = len(x)
    J = np.empty((m, m), dtype=float)
    for j in range(m):
        h = JACOBIAN_REL_STEP * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        fp, fm = F(xp), F(xm)
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


# Damped Newton solve from x0 using finite-difference Jacobians and least squares for
# singular systems. Return (point, residual, iterations), or None on failure.
def newton(F, x0):
    x = np.asarray(x0, dtype=float).copy()
    f = F(x)
    if not np.all(np.isfinite(f)):
        return None
    for iteration in range(MAX_ITER + 1):
        fnorm = float(np.linalg.norm(f, ord=np.inf))
        if fnorm <= RESIDUAL_TOL:
            return x, fnorm, iteration
        if iteration == MAX_ITER:
            break
        J = numerical_jacobian(F, x)
        if not np.all(np.isfinite(J)):
            return None
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            # Singular/near-singular Jacobian: least-squares Newton step.
            step, *_ = np.linalg.lstsq(J, -f, rcond=None)
        if not np.all(np.isfinite(step)):
            return None
        if np.linalg.norm(step, ord=np.inf) <= STEP_TOL * (1.0 + np.linalg.norm(x, ord=np.inf)):
            # Tiny step is accepted only if residual is already reasonably small.
            if fnorm <= max(RESIDUAL_TOL * 100, 1e-8):
                return x, fnorm, iteration
            return None

        # Damped Newton: require residual decrease when possible.
        alpha = 1.0
        accepted = False
        while alpha >= MIN_DAMPING:
            trial = x + alpha * step
            ft = F(trial)
            if np.all(np.isfinite(ft)) and np.linalg.norm(ft, ord=np.inf) < fnorm:
                x, f, accepted = trial, ft, True
                break
            alpha *= 0.5
        if not accepted:
            return None
    return None


# Build starting-point coordinate arrays from DEFAULT_GRID and per-variable overrides.
# These are Newton guesses, not the PDE's physical mesh.
def grid_axes(var):
    axes = []
    for i in var:
        lo, hi, count = GRID_OVERRIDES.get(i, DEFAULT_GRID)
        if not (math.isfinite(float(lo)) and math.isfinite(float(hi)) and float(lo) <= float(hi)):
            raise ValueError(f"Bad grid range for u[{i}]")
        if type(count) is not int or count < 1:
            raise ValueError(f"Grid count for u[{i}] must be positive integer")
        axes.append(np.linspace(float(lo), float(hi), count))
    total = math.prod(len(a) for a in axes)
    if total > MAX_GRID_POINTS:
        raise ValueError(f"Grid has {total:,} starting points > MAX_GRID_POINTS={MAX_GRID_POINTS:,}. "
                         "Reduce per-axis counts; Cartesian grids grow exponentially.")
    return axes, total


# Compare a candidate against accepted roots using the configured merge tolerance to avoid
# duplicate multistart results.
def is_same_root(a, b):
    scale = 1.0 + max(float(np.linalg.norm(a, ord=np.inf)), float(np.linalg.norm(b, ord=np.inf)))
    return float(np.linalg.norm(a - b, ord=np.inf)) <= ROOT_MERGE_TOL * scale


# Load the chosen equation/provider file, run the configured multistart search and write
# root CSV/NPZ files into output_dir.
def main(path: Path, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    source = path.read_text(encoding="utf-8-sig")
    order, defaults, compiled = load_equations(source)
    eq, var, params, F = make_problem(order, defaults, compiled)
    axes, total = grid_axes(var)

    print(f"Equations file: {path}")
    print("Equations:", ", ".join(f"du_dt[{i}] ({label(order, i)})" for i in eq))
    print("Unknowns: ", ", ".join(label(order, i) for i in var))
    print("Grid axes:")
    for i, axis in zip(var, axes):
        print(f"  u[{i}] {label(order, i)}: [{axis[0]:g}, {axis[-1]:g}], {len(axis)} points")
    print(f"Cartesian starts: {total:,}")

    roots = []
    hits = []
    failures = 0
    for k, start in enumerate(itertools.product(*axes), 1):
        result = newton(F, np.array(start, dtype=float))
        if result is None:
            failures += 1
            continue
        root, residual, iterations = result
        found = None
        for j, old in enumerate(roots):
            if is_same_root(root, old):
                found = j
                break
        if found is None:
            roots.append(root.copy())
            hits.append({"starts": 1, "best_residual": residual, "min_iterations": iterations})
        else:
            hits[found]["starts"] += 1
            hits[found]["best_residual"] = min(hits[found]["best_residual"], residual)
            hits[found]["min_iterations"] = min(hits[found]["min_iterations"], iterations)
        if total >= 1000 and (k % max(1, total // 20) == 0 or k == total):
            print(f"  {k:>{len(str(total))}}/{total} starts; unique roots so far: {len(roots)}")

    # Deterministic ordering for output.
    order_idx = sorted(range(len(roots)), key=lambda j: tuple(roots[j]))
    roots = [roots[j] for j in order_idx]
    hits = [hits[j] for j in order_idx]

    print(f"\nConverged starts: {total - failures:,}/{total:,}; unique roots: {len(roots)}")
    if roots:
        header = " | ".join(f"u[{i}]={label(order, i).split(' [')[0]}" for i in var)
        print(header + " | ||F||inf | starts | min_iter")
        for r, h in zip(roots, hits):
            vals = " | ".join(f"{x:.12g}" for x in r)
            print(f"{vals} | {h['best_residual']:.3e} | {h['starts']} | {h['min_iterations']}")

    out_csv = output_dir / OUTPUT_CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"u[{i}]_{label(order, i).split(' [')[0]}" for i in var]
                        + ["residual_inf", "converged_starts", "min_iterations"])
        for r, h in zip(roots, hits):
            writer.writerow(list(map(float, r)) + [h["best_residual"], h["starts"], h["min_iterations"]])

    out_npz = output_dir / OUTPUT_NPZ
    root_array = np.stack(roots) if roots else np.empty((0, len(var)))
    np.savez_compressed(out_npz,
        roots=root_array,
        equation_indices=np.array(eq, dtype=int),
        variable_indices=np.array(var, dtype=int),
        residual_inf=np.array([h["best_residual"] for h in hits], dtype=float),
        converged_starts=np.array([h["starts"] for h in hits], dtype=int),
        min_iterations=np.array([h["min_iterations"] for h in hits], dtype=int),
        parameters_json=np.array(json.dumps(params)),
        background_state_json=np.array(json.dumps(BACKGROUND_STATE)),
        order_json=np.array(json.dumps(order)),
        equations_path=np.array(str(path)))
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid multi-start Newton solver for selected Galerkin ODEs")
    parser.add_argument("--equations", type=Path, required=True, help="Path to equations.txt-like generated equations")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for analysis files")
    args = parser.parse_args()
    p = Path(args.equations or EQUATIONS_PATH).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    if not p.is_file():
        parser.error(f"Equations file not found: {p}")
    main(p, args.output)
