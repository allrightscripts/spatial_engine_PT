"""Refine one selected Galerkin equilibrium and compute the Jacobian spectrum.

Edit CONFIG and run:
    spatial-engine spectrum
or:
    spatial-engine spectrum --equations "equations.txt"

For DIMENSIONS = [i1, ..., iN], this script solves/refines
    du_dt[i1] = ... = du_dt[iN] = 0
with respect to
    u[i1], ..., u[iN],
then forms the N x N Jacobian and computes its eigenvalues/eigenvectors.

All coefficients outside DIMENSIONS are frozen at BACKGROUND_STATE (zero by default).
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
import numpy as np

# ============================== CONFIG ======================================
EQUATIONS_PATH = r"output/equations.txt"

# Numbers from the generated state-vector list.
# Example: [0, 5, 16] -> u_0_0, u_1_1, v_0_0.
DIMENSIONS = [0, 16]

# Coordinates of the point in EXACTLY the same order as DIMENSIONS.
# Example for DIMENSIONS=[0,16]: POINT=[u[0], u[16]]
POINT = [0.0, 0.0]

# Recommended: first refine POINT to an actual equilibrium by Newton,
# then compute the spectrum at the refined point.
REFINE_POINT_WITH_NEWTON = True

# Frozen values of all non-selected coefficients. Unlisted = 0.
BACKGROUND_STATE = {}

# Override parameters from equations.txt if needed.
PARAMETER_OVERRIDES = {}
T_VALUE = 0.0

# Numerical controls.
MAX_ITER = 80
RESIDUAL_TOL = 1e-10
STEP_TOL = 1e-12
JACOBIAN_REL_STEP = 1e-6
MIN_DAMPING = 2.0 ** -16

OUTPUT_CSV = "point_spectrum.csv"
OUTPUT_NPZ = "point_spectrum.npz"
# ===========================================================================

_NUMPY_NAMES = {"pi", "e", "sin", "cos", "tan", "exp", "log", "sqrt",
                "asin", "acos", "atan", "arcsin", "arccos", "arctan",
                "sinh", "cosh", "tanh", "floor", "ceil"}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
              ast.UAdd, ast.USub)


# Inspect the expression AST against the allowed syntax before compilation; this is the
# restricted legacy equation interface, not arbitrary Python execution.
def _check_expression(tree, allowed_names, size):
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
def load_equations(source):
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
def label(order, i):
    field, mode = order[i]
    return f"{field}_{'_'.join(map(str, mode))} [u[{i}]]"


# Build a callable subsystem using selected equations/variables and a frozen background
# state. Module settings determine which dimensions are analyzed.
def make_problem(order, defaults, compiled):
    n_total = len(order)
    dims = list(DIMENSIONS)

    if not dims:
        raise ValueError("DIMENSIONS cannot be empty")
    if len(set(dims)) != len(dims):
        raise ValueError("DIMENSIONS contains duplicates")
    if any(type(i) is not int or not 0 <= i < n_total for i in dims):
        raise ValueError(f"DIMENSIONS indices must be integers in 0..{n_total-1}")
    if len(POINT) != len(dims):
        raise ValueError("POINT must contain exactly one coordinate per DIMENSIONS entry")
    if set(BACKGROUND_STATE) & set(dims):
        raise ValueError("A selected DIMENSIONS entry cannot also be frozen in BACKGROUND_STATE")
    if any(type(i) is not int or not 0 <= i < n_total for i in BACKGROUND_STATE):
        raise ValueError("Invalid BACKGROUND_STATE index")

    params = {**defaults, **PARAMETER_OVERRIDES}
    unknown_params = set(PARAMETER_OVERRIDES) - set(defaults)
    if unknown_params:
        raise ValueError(f"Unknown parameter overrides: {sorted(unknown_params)}")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in params.values()):
        raise ValueError("Parameters must be finite numbers")

    base = np.zeros(n_total, dtype=float)
    for i, value in BACKGROUND_STATE.items():
        base[i] = float(value)

    safe_globals = {"__builtins__": {}}

    # Insert trial coordinates into a copy of the frozen background, evaluate selected RHS
    # components and return NaNs for invalid trial evaluations.
    def F(x):
        state = base.copy()
        state[dims] = np.asarray(x, dtype=float)
        ns = {"np": np, "t": float(T_VALUE), "u": state, **params}
        try:
            from engine.numerics.rhs import evaluate_rhs
            y = evaluate_rhs(compiled, float(T_VALUE), state, params)[dims]
        except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError):
            return np.full(len(dims), np.nan)
        return y

    return dims, params, F


# Approximate derivatives of F by centred finite differences with a scale-aware coordinate
# step; this is a local numerical Jacobian, not symbolic differentiation.
def numerical_jacobian(F, x):
    x = np.asarray(x, dtype=float)
    f0 = F(x)
    J = np.empty((len(f0), len(x)), dtype=float)
    for j in range(len(x)):
        h = JACOBIAN_REL_STEP * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        fp, fm = F(xp), F(xm)
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


# Refine a proposed point with damped Newton; return point/residual/iteration count or
# raise RuntimeError on nonconvergence/nonfinite data.
def newton(F, x0):
    x = np.asarray(x0, dtype=float).copy()
    f = F(x)
    if not np.all(np.isfinite(f)):
        raise RuntimeError("F(POINT) contains NaN/Inf")

    for iteration in range(MAX_ITER + 1):
        fnorm = float(np.linalg.norm(f, ord=np.inf))
        if fnorm <= RESIDUAL_TOL:
            return x, fnorm, iteration

        if iteration == MAX_ITER:
            break

        J = numerical_jacobian(F, x)
        if not np.all(np.isfinite(J)):
            raise RuntimeError("Jacobian contains NaN/Inf during Newton refinement")

        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(J, -f, rcond=None)

        if np.linalg.norm(step, ord=np.inf) <= STEP_TOL * (1.0 + np.linalg.norm(x, ord=np.inf)):
            if fnorm <= max(RESIDUAL_TOL * 100, 1e-8):
                return x, fnorm, iteration
            raise RuntimeError("Newton stagnated before reaching the requested residual")

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
            raise RuntimeError("Newton line search failed")

    raise RuntimeError(f"Newton did not converge in {MAX_ITER} iterations")


# Format complex eigenvalues consistently for human-readable reports, suppressing
# negligible imaginary parts.
def fmt_complex(z):
    z = complex(z)
    if abs(z.imag) < 1e-13:
        return f"{z.real:.12g}"
    return f"{z.real:.12g} {z.imag:+.12g}j"


# Load the selected subsystem, optionally refine its point and export the Jacobian
# eigenvalues/eigenvectors and analysis metadata.
def main(path: Path, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    source = path.read_text(encoding="utf-8-sig")
    order, defaults, compiled = load_equations(source)
    dims, params, F = make_problem(order, defaults, compiled)

    x_given = np.asarray(POINT, dtype=float)
    if not np.all(np.isfinite(x_given)):
        raise ValueError("POINT must contain finite numbers")

    print(f"Equations file: {path}")
    print(f"N = {dims}")
    print(f"Matrix size = {len(dims)} x {len(dims)}")
    print("Selected coordinates:")
    for k, i in enumerate(dims):
        print(f"  x[{k}] = {label(order, i)} = {x_given[k]:.12g}")

    given_residual = float(np.linalg.norm(F(x_given), ord=np.inf))
    print(f"\nResidual at supplied point ||F||_inf = {given_residual:.6e}")

    if REFINE_POINT_WITH_NEWTON:
        x, residual, iterations = newton(F, x_given)
        print(f"Newton refinement: converged in {iterations} iterations")
        print(f"Refined residual ||F||_inf = {residual:.6e}")
    else:
        x = x_given
        residual = given_residual
        iterations = 0
        print("Newton refinement: OFF")

    J = numerical_jacobian(F, x)
    eigenvalues, eigenvectors = np.linalg.eig(J)

    # Stable display order: descending real part, then descending imaginary part.
    order_idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
    eigenvalues = eigenvalues[order_idx]
    eigenvectors = eigenvectors[:, order_idx]

    print("\nPoint used for spectrum:")
    for k, i in enumerate(dims):
        print(f"  {label(order, i)} = {x[k]:.12g}")

    print("\nJacobian J =")
    print(np.array2string(J, precision=8, suppress_small=False))

    print("\nSpectrum:")
    for k, lam in enumerate(eigenvalues, start=1):
        stability = "stable" if lam.real < -1e-10 else ("unstable" if lam.real > 1e-10 else "neutral/near-zero")
        print(f"  lambda_{k} = {fmt_complex(lam)}    [{stability}]")

    n_stable = int(np.sum(eigenvalues.real < -1e-10))
    n_unstable = int(np.sum(eigenvalues.real > 1e-10))
    n_center = len(eigenvalues) - n_stable - n_unstable
    print(f"\nCounts: stable={n_stable}, unstable={n_unstable}, neutral/near-zero={n_center}")

    csv_path = output_dir / OUTPUT_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["N_indices", repr(dims)])
        w.writerow(["matrix_size", len(dims)])
        w.writerow(["newton_refined", REFINE_POINT_WITH_NEWTON])
        w.writerow(["newton_iterations", iterations])
        w.writerow(["residual_inf", residual])
        w.writerow([])
        w.writerow(["coordinate_k", "u_index", "label", "value"])
        for k, i in enumerate(dims):
            w.writerow([k, i, label(order, i), x[k]])
        w.writerow([])
        w.writerow(["eigenvalue_k", "real", "imag"])
        for k, lam in enumerate(eigenvalues, start=1):
            w.writerow([k, lam.real, lam.imag])

    npz_path = output_dir / OUTPUT_NPZ
    np.savez(
        npz_path,
        dimensions=np.asarray(dims, dtype=int),
        point_supplied=x_given,
        point_used=x,
        residual_inf=residual,
        newton_iterations=iterations,
        jacobian=J,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        labels=np.asarray([label(order, i) for i in dims]),
    )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--equations", required=True,
                        help="Path to generated equations.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for analysis files")
    args = parser.parse_args()
    p = Path(args.equations).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    if not p.is_file():
        parser.error(f"Equations file not found: {p}")
    main(p, args.output)
