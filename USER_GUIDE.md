# Spatial Engine User Guide

For **Spatial Engine v1.0**, Python **3.10 or newer**.

Spatial Engine simulates spatial fields in one, two or three dimensions. It uses a spectral Galerkin representation and saves a new segment when an event changes the system. You can inspect saved runs, export pictures and animations, examine field provenance, and compare a diffusion benchmark with an analytic solution.

This guide uses `python -m engine` throughout. The installed `spatial-engine` command is an equivalent shortcut, but the module form avoids common Windows PATH problems.

## 1. Install and check your environment

Unpack the release and open a terminal in the directory containing `pyproject.toml` and the `engine` folder.

Create a virtual environment:

```text
python -m venv .venv
```

Activate it using the command for your shell:

| Shell | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Linux or macOS | `source .venv/bin/activate` |

Install the package and check it:

```text
python -m pip install .
python -m engine --version
python -m engine --help
```

The version should be `1.0`. If PowerShell does not allow activation, you can use the environment's interpreter directly, for example:

```text
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\python.exe -m engine --version
```

Use that interpreter in place of `python` in subsequent commands. Installation and execution must use the same Python environment.

When upgrading, open a terminal in the **newly unpacked release directory** and install from there. An old `engine` folder in your current directory can shadow a newer installed package.

All relative paths below are relative to your terminal's current working directory. Quote paths containing spaces.

## 2. Run your first experiment

List the bundled examples:

```text
python -m engine examples
```

Export a configuration, run it, and save the first frame:

```text
python -m engine examples provenance_demo_1d --output experiment.json
python -m engine run --config experiment.json --output runs/experiment_01
python -m engine plot --manifest runs/experiment_01 --frame 0 --save runs/experiment_01/first.png
```

The provenance example starts with a T blob and demonstrates stochastic P births and T fragmentation. It is a useful starting point for understanding source and inheritance relationships.

Use a new or empty output directory for each run. The run command does not resume an existing experiment. The example exporter also refuses to overwrite an existing configuration; edit your exported JSON or choose another filename.

To view the saved run interactively:

```text
python -m engine plot --manifest runs/experiment_01
```

`--manifest` accepts the run directory or its `manifest.json` path. Plotting reads the archive; it does not rerun the simulation.

### Choose an example

| Example | Use |
|---|---|
| `provenance_demo_1d` | Small source/inheritance and fragmentation demonstration |
| `basic_2d` | Small two-dimensional simulation |
| `basic_3d` | Small three-dimensional simulation |
| `sandbox_3d_per_T` | Larger three-dimensional cultural scenario with concrete T sources |
| `spectral_periodic_3d` | Periodic spectral example |
| `finite_volume_3d` | Configuration for the separate finite-volume backend |
| `reference_diffusion` | Parameters for the one-dimensional diffusion benchmark |
| `reference_diffusion_3d` | Parameters for a three-dimensional diffusion benchmark |
| `reference_diffusion_smooth` | Smooth multimode reference for convergence experiments |

The `reference_*` files are **reference parameters**, not full cultural run configurations. Use them with `diffusion`, as shown below. The finite-volume example uses the separate `finite-volume` command; check its `--help` before running it.

For a small 2D run:

```text
python -m engine examples basic_2d --output sandbox_2d.json
python -m engine run --config sandbox_2d.json --output runs/sandbox_2d
```

## 3. Save a picture at a particular time

Use `--time` to select a saved frame by model time:

```text
python -m engine plot --manifest runs/sandbox_2d/manifest.json --time 0.5 --save runs/sandbox_2d/frame_t0_5.png
```

Choose a time within your run's saved range. For example, `--time 12.5` is valid only if the archive covers that time. Use a decimal point, not a decimal comma.

Time is expressed in the model's units. It represents seconds only if your model uses seconds.

**The command selects an existing saved frame. It does not interpolate or integrate to the requested time.** It reports the requested time, actual saved time, frame index and segment. The figure title uses the actual time.

### Select which saved time to use

| Option | Behavior |
|---|---|
| `--time-mode nearest` | Closest saved time; equal-distance ties select the earlier time. This is the default. |
| `--time-mode before` | Latest saved time less than or equal to the requested time |
| `--time-mode after` | Earliest saved time greater than or equal to the requested time |

For example:

```text
python -m engine plot --manifest runs/sandbox_2d --time 0.5 --time-mode before --save runs/sandbox_2d/before_t0_5.png
```

Times outside the saved range, NaN and infinity produce an error rather than silently selecting an endpoint. To obtain more closely spaced frames in a future run, reduce `save_dt` in its configuration. This increases archive size.

### Select a side of an event

An event can produce multiple saved states at the same time: before and after the system changes. Use:

- `--event-side after`: choose the last frame at that saved time in manifest order; this is the default.
- `--event-side before`: choose the first frame at that saved time.

```text
python -m engine plot --manifest runs/experiment_01 --time 0.5 --event-side before --save runs/experiment_01/event_before.png
python -m engine plot --manifest runs/experiment_01 --time 0.5 --event-side after --save runs/experiment_01/event_after.png
```

These options only show an event jump when the chosen saved time actually has multiple states. `--time-mode before` includes an exact time match; it does **not** by itself select the pre-event state. Use `--event-side before` for that purpose.

If several events share a timestamp and you need an intermediate state, select its explicit frame index.

### Select by frame index

```text
python -m engine plot --manifest runs/sandbox_2d --frame 0 --save runs/sandbox_2d/first.png
```

Frame indices are zero-based. You cannot combine `--frame` and `--time`. The `--time-mode` and `--event-side` options require `--time`.

Without `--save`, a selected frame opens in a viewer. Without either `--frame` or `--time`, the viewer animates the run. Saving creates missing parent directories and can replace an existing image at the destination, so use distinct filenames for pictures you want to retain.

## 4. Export animations and inspect 3D runs

Create a GIF:

```text
python -m engine video --manifest runs/experiment_01 --output runs/experiment_01/movie.gif --fps 12
```

`--fps` controls playback speed, not simulated time or numerical accuracy. GIF export uses Pillow. MP4 export additionally requires FFmpeg on PATH; use the video command's `--ffmpeg` option to provide its executable path when necessary. Check `python -m engine video --help` for overwrite and other export options.

For 3D:

```text
python -m engine examples basic_3d --output experiment3d.json
python -m engine run --config experiment3d.json --output runs/experiment3d
python -m engine plot --manifest runs/experiment3d --frame 0 --slices 0.25 0.5 0.75 --grid 30 --save runs/experiment3d/slices.png
```

The 3D viewer displays orthogonal slices. `--slices` specifies relative slice positions. The plotting `--grid` option controls reconstruction for display, not the simulation's resolution. Use `--colorscale global` when you need fixed color ranges across frames; finding global ranges requires scanning saved data.

On a machine without a graphical desktop, use image or video export rather than an interactive window.

## 5. Edit a configuration safely

Start from an exported example. JSON does not allow comments or trailing commas. Keep notes in a separate text file or in your experiment records.

| Setting | Meaning |
|---|---|
| `seed` | Seed for stochastic choices; retain it when comparing configurations |
| `dimension` | Number of spatial axes: 1, 2 or 3 |
| `grid.shape` | Number of physical grid cells along each axis |
| `grid.lengths` | Physical domain length along each axis |
| `t_end` | Final model time |
| `save_dt` | Interval between requested saved samples, not the integrator's internal step |
| `dt` | Finite-volume time step; also used by spectral defaults where applicable |
| `solver.modes` | Spectral resolution along each axis |
| `solver.integrator` | Time integration method, such as `DOP853`, `Radau` or `BDF` |
| `solver.rtol`, `solver.atol` | Time integrator error tolerances |
| `solver.max_step` | Maximum internal integration step |
| `initial_fields` | Initial field IDs, families, model types, geometry and parameters |
| `events` | Scheduled events |
| `split` | Fragmentation detection settings, including threshold and checking cadence |

For sine/cosine bases, modes specify retained mode counts. For the Fourier basis, they specify frequency cutoffs, giving `2*k + 1` functions per axis for cutoff `k`. Do not assume every basis has the same coefficient count.

The lengths of `grid.shape`, `grid.lengths`, `solver.modes` and spatial coordinate arrays must agree with `dimension`. Geometry centers, radii and axis lengths use physical domain coordinates.

An initial field's amplitude comes from its initial value and geometry. Its `family` identifies its role and provenance; it does not automatically choose a reaction law. Specify an appropriate `model_type`, such as `tradition_v1` or `progressive_v1`, when that law is intended. An omitted model type uses the default behavior. See `REACTION_LAWS` in `engine/model/cultural_v01.py`.

Start with small configurations. The number of coefficients grows roughly as the number of active fields multiplied by the product of basis sizes across axes. A large 3D basis combined with many births can become expensive quickly. Tightening time tolerances does not replace a spatial convergence study.

## 6. Understand P birth and provenance

Three different mechanisms answer three different questions:

| Question | Where it is configured or implemented |
|---|---|
| When does a new P appear? | Stochastic rate settings and `engine/model/event_rules.py` |
| What spatial profile does it start with? | `birth_profile` and `engine/model/birth_profiles.py` |
| What happens after birth? | Continuous dynamics in `engine/model/cultural_v01.py` |

The recommended spatial-source settings under `stochastic_events.p_birth` are:

```json
{
  "source_mode": "per_T",
  "birth_profile": "inside_T",
  "birth_profile_parameters": {
    "amplitude": 1.0,
    "power": 3.0,
    "epsilon": 1e-12
  }
}
```

This is a configuration fragment, not a complete runnable file. Keep the example's enabled flag, rate settings and parameter policies around it.

In `per_T` mode, each active T has a nonnegative hazard. The total hazard sets the birth clock. When a birth occurs, a concrete source is chosen with probability equal to its hazard divided by the total hazard.

The `inside_T` profile uses only that selected source:

```text
P_new(x) = amplitude * [max(T_source(x), 0) /
                       (max_x max(T_source(x), 0) + epsilon)] ** power
```

The existing landscape mechanism is then applied. Larger powers concentrate the initial profile in stronger parts of the selected T field. Explicit profile amplitude overrides the legacy birth amplitude setting.

For compatibility, an omitted `source_mode` retains `global_T_sum` behavior. In that mode, `inside_T` uses the sum of T fields. An omitted birth profile retains uniform initialization. In per-T mode, an unusably small source is handled explicitly rather than silently introducing a uniform birth outside the selected source.

Spatial source and parameter inheritance are independent:

- `source_T_id` identifies where P originated.
- `inheritance_parent_id` identifies the P supplying inherited parameters, if any.
- `lineage_path` and `display_label` describe readable genealogy.
- Stable field IDs remain the archive identities.

One root T can produce `P_root;0`, `P_root;1` and `P_root;2`. These are source siblings, not automatically a P inheritance chain. After T splits into `T_0` and `T_1`, later births can be labelled `P_0;0` and `P_1;0`. Parameter inheritance can cross those source branches.

The first P has no inheritance parent and uses the configured parameter fallback. `inheritance_parent_policy` controls later parent selection; legacy `parent_policy` remains an alias. Parameter fallback and zero-profile fallback are separate settings with separate meanings. Existing P fields retain their historical source when that T later splits.

See [provenance and event semantics](docs/PROVENANCE_0_9.md) for the full schema and policies.

## 7. Target events explicitly

Legacy `targets` contain exact stable field IDs. They do not automatically mean a whole family or all descendants.

For a scheduled catastrophe affecting all active T fields, add an entry like this to the configuration's `events` list:

```json
{
  "time": 0.5,
  "type": "catastrophe",
  "strength": 0.5,
  "duration": 0,
  "effect_type": "impulse",
  "target_selector": {
    "mode": "family",
    "families": ["T"]
  }
}
```

Choose a time within the simulation interval. With no restricted geometry, this impulse applies across the domain and halves the selected fields.

Other semantic selectors include `{"mode": "all_active"}` and family selectors with `"families": ["T", "P"]`. Lineage-subtree targeting is described in the provenance guide. Use one targeting approach per event rather than mixing explicit targets and a selector.

Inspect `resolved_target_ids` in event metadata to see which fields an event actually affected. Human-readable labels are display metadata, not substitutes for stable IDs.

## 8. Read the archive and export genealogy

A run contains its manifest, configuration, recorded model source and numbered segment directories. A segment contains its trajectory and the information needed to interpret its fixed system.

Segments are not individual frames. Events can change the field set or right-hand side, so coefficient layouts may differ between segments. Do not concatenate raw coefficient columns without using the archived field and mode mappings. Move or share the entire run directory so referenced files stay together.

Export genealogy:

```text
python -m engine genealogy --manifest runs/experiment_01 --csv runs/experiment_01/genealogy.csv --graphviz runs/experiment_01/genealogy.dot --output runs/experiment_01/genealogy.txt
```

The analyzer presents source provenance separately from P parameter inheritance. CSV is useful for tables and external plotting. DOT export does not require Graphviz; rendering a DOT file is a separate step using Graphviz or another compatible viewer.

For archive details, see the [archive contract](docs/ARCHIVE_CONTRACT.md).

## 9. Validate against an exact diffusion solution

Keep the numerical benchmark separate from the cultural scenario:

```text
python -m engine examples reference_diffusion --output reference.json
python -m engine diffusion --reference-params reference.json --modes 8 --output runs/check_diffusion
python -m engine analyze --segment runs/check_diffusion/segment_000 --reference diffusion_neumann_cosine --reference-params reference.json --field u --output runs/check_diffusion/errors.csv
```

The analyzer compares saved frames with the reference and exports:

| Metric | Meaning |
|---|---|
| `L2` | Square root of the volume-weighted sum of squared errors |
| `rel_L2` | L2 error divided by reference L2 norm plus epsilon |
| `Linf` | Maximum absolute pointwise error |
| `mass_abs` | Absolute difference between numerical and reference integrals |
| `mass_rel` | Mass error divided by absolute reference mass plus epsilon |

Grid volume matters: L2 is not an unweighted Euclidean norm. Relative errors can be large when the reference norm or mass is near zero; inspect absolute errors as well.

For convergence studies, keep the physical problem, reference parameters and comparison times fixed while changing spectral resolution. Save each run separately. A single cosine may already be represented exactly by a small basis, so use the smooth multimode benchmark when studying truncation error. See [numerical validation](docs/VALIDATION.md) for worked experiments.

Do not compare a cultural reaction/event scenario against a pure diffusion formula unless that formula is actually a solution of the configured problem.

### Advanced segment analysis

Without a reference argument, the analyzer can perform Newton/spectrum analysis:

```text
python -m engine analyze --segment runs/experiment_01/segment_000 --output runs/experiment_01/point_spectrum.npz
```

This is a different operation from reference-error measurement. Newton convergence is not guaranteed. Nonautonomous systems require an explicit frozen analysis time where supported; check `analyze --help`.

Analysis that evaluates the archived model requires a matching model source hash. Even comment-only edits can change a byte-based source hash. For older runs, restore their recorded source in a separate checkout when necessary. Saved-field visualization and reference-error reconstruction do not require evaluating a modified cultural model.

## 10. Troubleshooting

| Symptom | What to check |
|---|---|
| `spatial-engine` is not recognized | Use `python -m engine`; install with that same interpreter. |
| `No module named engine` | Activate the intended environment and run `python -m pip install .` from the project root. |
| `--time` is not recognized | Check `--version`, install v1.0 and leave any old source directory that shadows it. |
| Output directory already contains files | Choose a new directory. Existing runs are not resumed. |
| Requested time is outside the archive | Use the reported saved bounds; a longer run must be simulated separately. |
| No P births occurred | Check enabled status, birth limit, rates and run duration. A stochastic run can legitimately contain no birth. |
| Per-T source is too small | Inspect source magnitude, epsilon and the source rate rule rather than masking the issue with a global uniform profile. |
| Dimension or shape error | Check all axis-dependent arrays and geometry coordinates. |
| A 3D run is slow | Start with `basic_3d`; reduce modes, grid size, duration or birth count as appropriate. |
| Oscillations or negative reconstructed values | Examine basis resolution and initial-profile smoothness; perform a convergence study. |
| Model source hash mismatch | Use the recorded matching source for model-dependent analysis. |
| MP4 export fails | Check FFmpeg availability, or export GIF. |

For diagnostic output:

```text
python -m engine run --config experiment.json --output runs/debug_01 --debug
```

When reporting a problem, include the version, exact command, configuration and complete traceback. Preserve the seed and dependency versions for reproducibility. Identical seeds do not promise bit-for-bit results across all platforms and numerical library versions.

## 11. Modify the code and contribute

Use an editable installation when changing the implementation:

```text
python -m pip install -e ".[test]"
python -m unittest discover -s tests -q
```

Read the existing docstrings and comments before editing. Keep changes focused and preserve stable IDs, archive mappings and the distinction between source and inheritance. Do not rename the `engine` package casually: archived descriptors refer to its import paths.

Useful entry points:

- [Function reference](docs/FUNCTION_REFERENCE.md): runtime functions, signatures and descriptions.
- [Developer guide, in Russian](docs/EXTENDING.md): where to change reactions, hazards, birth profiles and related behavior.
- [Migration notes](docs/MIGRATION.md): compatibility across releases.
- [Release checks](docs/RELEASE_CHECKS.md): checks before distributing a modified version.

Use `python -m engine <command> --help` for the exact options supported by your installed version.
