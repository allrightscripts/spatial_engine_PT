"""Generate explicit cosine Galerkin equations for homogeneous Neumann BC.

Edit MODEL below, then run: python PDE_to_ODE.py --output equations.txt
Requirement: sympy. Polynomial field nonlinearities use an exact sparse
cosine convolution with a persistent interaction cache. Output contains NumPy
expressions, not an executable function.
"""

from __future__ import annotations

import argparse
import re
from itertools import product
from collections import Counter
import pickle
from pathlib import Path

import sympy as sp


DIMENSIONS = 2
MODE_RANGE = [0, 3]
MODEL = {
    "dimensions": DIMENSIONS,
    "mode_ranges": (MODE_RANGE,) * DIMENSIONS,
    "fields": ("u", "v"),
    "parameters": {"Du": 0.1, "Dv": 0.2, "a": 1.0, "b": 0.5,
                   **{f"Lx{i}": 1.0 for i in range(1, DIMENSIONS + 1)}},
    "rhs": {
        "u": "Du*lap(u) + a*u - b*u*v",
        "v": "Dv*lap(v) + b*u*v - v",
    },
}


def generate(model: dict) -> str:
    dimensions = model.get("dimensions", len(model.get("axes", ())))
    if type(dimensions) is not int or not 1 <= dimensions <= 3:
        raise ValueError("dimensions must be 1, 2 or 3")
    axes = tuple(model.get("axes", (f"x{i}" for i in range(1, dimensions + 1))))
    lengths = tuple(model.get("lengths", (f"Lx{i}" for i in range(1, dimensions + 1))))
    mode_ranges = tuple(model["mode_ranges"])
    fields = tuple(model["fields"])
    parameters = dict(model["parameters"])
    rhs_strings = dict(model["rhs"])
    if not (dimensions == len(axes) == len(lengths) == len(mode_ranges)):
        raise ValueError("dimensions must match the number of axes, lengths and mode ranges")
    if not fields or len(set(fields)) != len(fields) or set(fields) != set(rhs_strings):
        raise ValueError("Each distinct field needs exactly one RHS")
    if any(len(bounds) != 2 or any(type(k) is not int for k in bounds)
           or bounds[0] < 0 or bounds[0] > bounds[1] for bounds in mode_ranges):
        raise ValueError("Each mode range must be [k1, kn] with 0 <= k1 <= kn; negative cosine modes duplicate positive ones")
    names = ("t",) + axes + fields + tuple(parameters)
    if len(set(names)) != len(names):
        raise ValueError("Axes, lengths, fields and parameters must have distinct names")
    if not set(lengths) <= set(parameters):
        raise ValueError("Every length must have a default value in parameters")
    if any(not name.isidentifier() for name in names):
        raise ValueError("All names must be Python identifiers")

    xs = sp.symbols(" ".join(axes), real=True)
    ls = sp.symbols(" ".join(lengths), positive=True, real=True)
    xs = (xs,) if len(axes) == 1 else xs
    ls = (ls,) if len(axes) == 1 else ls
    time = sp.Symbol("t", real=True)
    p = {name: sp.Symbol(name, real=True) for name in parameters}
    p.update(zip(lengths, ls))
    # Keep fields abstract while differentiating the PDE. This avoids expanding
    # the Fourier series before the sparse Galerkin projection.
    funcs = {f: sp.Function(f)(*xs) for f in fields}

    def divergence(*components):
        if len(components) != dimensions:
            raise ValueError(f"div expects {dimensions} components, one per axis")
        return sum(sp.diff(component, x) for component, x in zip(components, xs))

    scope = {"t": time, **dict(zip(axes, xs)), **p, **funcs,
             "diff": sp.diff, "sin": sp.sin, "cos": sp.cos,
             "exp": sp.exp, "pi": sp.pi,
             "lap": lambda field: sum(sp.diff(field, x, 2) for x in xs),
             "div": divergence}
    pdes = {}
    for f in fields:
        expression = sp.sympify(rhs_strings[f], locals=scope)
        unknown = expression.free_symbols - set(p.values()) - set(xs) - {time}
        if unknown:
            raise ValueError(f"Unknown symbols in RHS of {f}: {unknown}")
        unknown_functions = {a for a in expression.atoms(sp.Function)
                             if a not in funcs.values() and not a.has(*funcs.values())}
        if unknown_functions:
            raise ValueError(f"Unknown functions in RHS of {f}: {unknown_functions}")
        pdes[f] = sp.expand(expression)

    indices = list(product(*(range(k1, kn + 1) for k1, kn in mode_ranges)))
    q = {(f, idx): sp.Symbol(f"q_{f}_{'_'.join(map(str, idx))}") for f in fields for idx in indices}

    # ------------------------------------------------------------------
    # Sparse tensor Galerkin engine
    # ------------------------------------------------------------------
    # We NEVER expand (sum q_i phi_i)^p.  Instead each local PDE monomial is
    # contracted directly with a cached basis-interaction tensor.  A tensor
    # entry is an exact product of 1D trigonometric integrals.  Selection rules
    # make most entries exactly zero, and zeros are cached too.
    cache_dir = Path(model.get("cache_dir") or (Path(__file__).resolve().parent / ".galerkin_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_cache_file = cache_dir / f"cosine_{dimensions}d_tensor.pkl"
    try:
        with tensor_cache_file.open("rb") as fh:
            tensor_cache = pickle.load(fh)
    except Exception:
        tensor_cache = {}
    cache_start_size = len(tensor_cache)

    # Replace fields/derivatives by algebraically independent jet symbols.
    # This lets SymPy expand the *small PDE formula*, not the Fourier series.
    jet_symbol = {}
    jet_info = {}
    def jet(f, orders):
        key = (f, tuple(orders))
        if key not in jet_symbol:
            tag = "_".join(map(str, orders))
            z = sp.Symbol(f"__J_{f}_{tag}")
            jet_symbol[key] = z
            jet_info[z] = key
        return jet_symbol[key]

    def derivative_orders(d):
        c = Counter(d.variables)
        return tuple(c[x] for x in xs)

    jet_pdes = {}
    for f, expr in pdes.items():
        repl = {}
        # Derivatives first; xreplace is simultaneous, so no nested ambiguity.
        for d in expr.atoms(sp.Derivative):
            for fn in fields:
                if d.expr == funcs[fn]:
                    repl[d] = jet(fn, derivative_orders(d))
                    break
        for fn in fields:
            repl[funcs[fn]] = jet(fn, (0,)*dimensions)
        jet_pdes[f] = sp.expand(expr.xreplace(repl))

    def one_axis_product_integral(freqs, deriv_orders, L):
        """Exact integral of product of derivatives of cosine modes on [0,L]."""
        # d^r cos(n*pi*x/L) = (n*pi/L)^r cos(n*pi*x/L + r*pi/2).
        amp = sp.Integer(1)
        phases = []
        for n, r in zip(freqs, deriv_orders):
            if r and n == 0:
                return sp.Integer(0)
            amp *= (sp.pi*n/L)**r if r else 1
            phases.append(sp.Rational(r, 2)*sp.pi)
        m = len(freqs)
        # exponential/cosine product selection rule.  Integrals vanish unless
        # a signed sum of wave numbers is zero.
        acc = sp.Integer(0)
        for signs in product((-1, 1), repeat=m):
            if sum(s*n for s, n in zip(signs, freqs)) == 0:
                acc += sp.cos(sum(s*ph for s, ph in zip(signs, phases)))
        return amp * L * acc / (2**m)

    def tensor_entry(test_idx, factors):
        # factors = ((mode_idx, derivative_orders), ...). Test function has r=0.
        factors = tuple(sorted(factors, key=repr))
        key = (test_idx, factors)
        if key in tensor_cache:
            return tensor_cache[key]
        val = sp.Integer(1)
        for ax, L in enumerate(ls):
            freqs = [test_idx[ax]] + [idx[ax] for idx, _ in factors]
            ders = [0] + [orders[ax] for _, orders in factors]
            val *= one_axis_product_integral(freqs, ders, L)
            if val == 0:
                break
        val = sp.cancel(val)
        tensor_cache[key] = val
        return val

    axis_selection_cache = {}

    def project_tensor(expr, test_idx):
        norm = sp.prod(L/(2 if n else 1) for n, L in zip(test_idx, ls))
        total = sp.Integer(0)
        for term in sp.Add.make_args(sp.expand(expr)):
            powers = term.as_powers_dict()
            jets = []
            coeff = term
            valid = True
            for z, power in list(powers.items()):
                if z not in jet_info:
                    continue
                if not power.is_Integer or int(power) < 0:
                    valid = False
                    break
                power = int(power)
                coeff /= z**power
                jets.extend([jet_info[z]] * power)
            if not valid or any(z in coeff.free_symbols for z in jet_info):
                raise ValueError(
                    "Tensor engine requires polynomial dependence on fields and their "
                    f"spatial derivatives. Unsupported term: {term}"
                )
            if coeff.has(*xs, time) or coeff.has(sp.Derivative):
                raise ValueError(
                    "Sparse tensor projection requires spatially and temporally "
                    f"constant polynomial coefficients. Unsupported term: {term}"
                )
            if not jets:
                # Constant-in-space source projects only onto the zero mode.
                integ = sp.prod(L if n == 0 else 0 for n, L in zip(test_idx, ls))
                total += coeff * integ
                continue
            # Generate only mode tuples allowed by the cosine-product selection
            # rules.  For every spatial axis, a nonzero integral requires a
            # signed sum of the test frequency and factor frequencies to vanish.
            # We solve that constraint axis-by-axis and combine the surviving
            # coordinates, instead of scanning |I|**p full mode tuples.
            degree = len(jets)

            def axis_candidates(ax):
                lo, hi = mode_ranges[ax]
                ders = tuple(orders[ax] for _, orders in jets)
                key = (ax, test_idx[ax], ders)
                if key in axis_selection_cache:
                    return axis_selection_cache[key]
                out = []
                # Choose signs for test + all factors. Overall sign reversal is
                # redundant, so fix the test sign to +1. Then solve for the last
                # factor frequency rather than enumerate it.
                if degree == 1:
                    sign_patterns = [()]
                else:
                    sign_patterns = tuple(product((-1, 1), repeat=degree-1))
                seen = set()
                for prefix in product(range(lo, hi+1), repeat=max(0, degree-1)):
                    for signs_prefix in sign_patterns if degree > 1 else [()]:
                        partial = test_idx[ax] + sum(s*n for s, n in zip(signs_prefix, prefix))
                        # Last factor sign can be +/-1. Required n_last is abs(partial).
                        nlast = abs(partial)
                        if not (lo <= nlast <= hi):
                            continue
                        tup = tuple(prefix) + (nlast,)
                        # Derivative of the zero cosine mode vanishes.
                        if any(r and n == 0 for n, r in zip(tup, ders)):
                            continue
                        if tup in seen:
                            continue
                        # Phase/parity may still make an otherwise frequency-allowed
                        # interaction zero; check the cheap exact 1D integral once.
                        freqs = [test_idx[ax]] + list(tup)
                        dord = [0] + list(ders)
                        if one_axis_product_integral(freqs, dord, ls[ax]) != 0:
                            seen.add(tup)
                            out.append(tup)
                axis_selection_cache[key] = tuple(out)
                return axis_selection_cache[key]

            axis_lists = [axis_candidates(ax) for ax in range(dimensions)]
            for coords_by_axis in product(*axis_lists):
                chosen = tuple(
                    tuple(coords_by_axis[ax][factor] for ax in range(dimensions))
                    for factor in range(degree)
                )
                factors = tuple((idx, orders) for idx, (_, orders) in zip(chosen, jets))
                coupling = tensor_entry(test_idx, factors)
                if coupling == 0:
                    continue
                state_prod = sp.prod(q[fld, idx] for idx, (fld, _) in zip(chosen, jets))
                total += coeff * coupling * state_prod
        return sp.cancel(total / norm)

    equations = []
    for f in fields:
        for idx in indices:
            equations.append(project_tensor(jet_pdes[f], idx))

    ordering = [(f, idx) for f in fields for idx in indices]
    slot = {q[key]: sp.Symbol(f"u[{i}]") for i, key in enumerate(ordering)}
    def mode_label(index):
        return "_".join(map(str, index))

    lines = ["# Cosine Galerkin equations; homogeneous Neumann basis.",
             "# Domain: " + ", ".join(f"{axis} in [0, {length}]"
                                       for axis, length in zip(axes, lengths)),
             "# Included modes: " + ", ".join(f"{axis}: [{k1}, {kn}]"
                                               for axis, (k1, kn) in zip(axes, mode_ranges)),
             f"ORDER = {ordering!r}",
             f"DEFAULT_PARAMETERS = {parameters!r}", "",
             "# State vector: each u[i] is a coefficient of the field shown below."]
    for i, (field, index) in enumerate(ordering):
        lines.append(f"# u[{i}] = {field}_{mode_label(index)}  (field {field}, mode {index})")
    lines.append("")
    for i, expr in enumerate(equations):
        # Keep expressions expanded in full: no temporary common subexpressions.
        code = sp.pycode(expr.xreplace(slot))
        names = set(re.findall(r"\bmath\.(\w+)", code))
        unsupported = names - {"pi", "e", "sin", "cos", "tan", "exp", "log", "sqrt",
                               "asin", "acos", "atan", "sinh", "cosh", "tanh", "floor", "ceil"}
        if unsupported:
            raise ValueError(f"No direct NumPy replacement for math functions: {sorted(unsupported)}")
        field, index = ordering[i]
        if i == 0 or ordering[i - 1][0] != field:
            lines.append(f"# Field {field}")
        lines.append(f"# d({field}_{mode_label(index)})/dt, mode {index}")
        lines.append(f"du_dt[{i}] = {code.replace('math.', 'np.')}")
        if i + 1 < len(ordering) and ordering[i + 1][0] != field:
            lines.append("")
    lines.append("")
    # Do not retain a partial cache when generating the equations fails.
    with tensor_cache_file.open("wb") as fh:
        pickle.dump(tensor_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write generated ODE to this path")
    args = parser.parse_args()
    code = generate(MODEL)
    if args.output:
        args.output.write_text(code, encoding="utf-8")
        print(f"Generated {args.output} ({len(MODEL['fields'])} fields)")
    else:
        print(code)
