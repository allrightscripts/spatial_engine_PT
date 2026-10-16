# Numerical validation

The reference registry is `engine/analysis/reference_solutions.py`. It supports
`diffusion_neumann_cosine` in 1D/2D/3D. For cosine modes k_i on lengths L_i,

    u(x,t) = exp[-D*pi²*sum_i(k_i/L_i)²*t] * product_i cos(k_i*pi*x_i/L_i).

`reference_diffusion_smooth` supplies a multi-mode example for resolution studies.
Copy configs using `spatial-engine examples <name> --output <file.json>`.

For discrete error e = u_num-u_ref and grid cell volume V:

    L2       = sqrt(sum |e|² V)
    rel_L2   = L2 / (sqrt(sum |u_ref|² V) + epsilon)
    Linf     = max |e|
    mass_abs = |sum u_num V - sum u_ref V|
    mass_rel = mass_abs / (|sum u_ref V| + epsilon)

Use `spatial-engine diffusion` to initialize the exact benchmark correctly;
running its archived config directly does not replace benchmark initialization.
Then use `spatial-engine analyze --reference diffusion_neumann_cosine` on each
saved segment. CSV/JSON rows contain time, field, L2, rel_L2, Linf, mass_abs and
mass_rel. Keep the reference, physical grid, times, tolerances and seed fixed
when studying mode counts. An exactly represented single cosine may already
reach the time-integrator error floor at low resolution.

For 3D:

```bash
spatial-engine examples reference_diffusion_3d --output reference3d.json
spatial-engine diffusion --reference-params reference3d.json --modes 4 4 4 --grid 12 14 16 --lengths 1 1.2 1.4 --output runs/diffusion3d
spatial-engine analyze --segment runs/diffusion3d/segment_000 --reference diffusion_neumann_cosine --reference-params reference3d.json --field u --output runs/diffusion3d/errors.csv
```

`analyze` without a reference performs Newton and local spectrum analysis; it
requires an explicit output path. For a nonautonomous model, supply `--time` and
interpret the result as frozen-time analysis, not a flow equilibrium.
