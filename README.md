# Spatial Engine

**Version 1.0 · research software · Python 3.10+**

Simulation of spatial fields in 1D, 2D and 3D using a spectral Galerkin
representation. Birth, fragmentation and external events can change the number
of fields during a run. Each interval with a fixed system is archived separately;
plots and analyses read those archives without repeating the simulation.

The package includes per-T stochastic P birth, separate spatial-source and
parameter-inheritance genealogy, semantic event selectors, and validation
against analytic diffusion solutions.

## Documentation

- [User guide](USER_GUIDE.md): installation, configuration, commands, saved-time selection and troubleshooting.
- [Extending the model](docs/EXTENDING.md): reactions, hazards, birth profiles, masks, reference solutions and invariants.
- [Function reference](docs/FUNCTION_REFERENCE.md): documented runtime functions and interfaces.

## Manuscript and reproducibility

This repository provides the software accompanying the manuscript:

*Oleh Podpriatov (2026). Modular spectral system for spatial dynamics with event-driven dimension change.*

The representative two-dimensional experiment is configured in
[`paper/representative_2d.json`](paper/representative_2d.json). This file matches
`global_config` in the run manifest used for the manuscript, including the
pseudorandom seed `2718`. The reported experiment is **2D**; the engine also
provides 1D and 3D configurations.

From an installed checkout, reproduce the numerical run with:

```bash
python -m engine run --config paper/representative_2d.json --output runs/paper_2d
```

The model generates a segmented run archive with a `manifest.json`. The same
seed and configuration are intended for reproducing the reported event sequence;
numerical details may depend on the installed Python and scientific-library
versions. See [numerical validation](docs/VALIDATION.md) and
[release checks](docs/RELEASE_CHECKS.md).

The manuscript-associated snapshot is intended to be preserved under the Git
tag `v1.0-paper`. Add the arXiv link here once the preprint is public.

## Install

From the unpacked project directory:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install .
python -m engine --version
```

The installed `spatial-engine <command>` executable is an equivalent shortcut.
The examples use `python -m engine` to avoid Windows PATH issues. No source edits or PYTHONPATH changes
are needed to run the installed package from another directory.

## Save a frame at a chosen model time

```powershell
python -m engine plot --manifest runs/sandbox_2d/manifest.json --time 12.5 --save runs/sandbox_2d/frame_t12_5.png
```

Selects the nearest **saved** frame and prints its actual time; no interpolation
or recomputation. `--time-mode before|after` selects a saved time on one side;
`--event-side before|after` resolves duplicate event-boundary times (default after).
Times outside the saved range fail explicitly. `--frame N` remains supported.
See the [user guide](USER_GUIDE.md) for full selection semantics.

## First run

Copy a bundled configuration to an editable file, then run it:

```bash
python -m engine examples provenance_demo_1d --output demo.json
python -m engine run --config demo.json --output runs/demo
python -m engine plot --manifest runs/demo --frame 0 --save runs/demo/frame.png
python -m engine genealogy --manifest runs/demo --csv runs/demo/genealogy.csv --graphviz runs/demo/genealogy.dot
```

The output directory must be empty. Change parameters in `demo.json` and use a
new output directory for each experiment. The example starts with one T blob;
P fields are born from concrete T sources, and subsequent fragmentation creates
new source branches. Inheritance is recorded separately.

For a 2D example, export `basic_2d` using the same command.

## 3D

```bash
python -m engine examples basic_3d --output demo3d.json
python -m engine run --config demo3d.json --output runs/demo3d
python -m engine plot --manifest runs/demo3d
python -m engine video --manifest runs/demo3d --output runs/demo3d/movie.gif
```

The 3D viewer uses orthogonal slices. `--slices 0.5 0.5 0.5` sets their relative
positions. GIF uses Pillow; MP4 additionally needs FFmpeg on PATH. Plot/video
commands use saved data only. For a larger cultural scenario, export
`sandbox_3d_per_T`; it is substantially more expensive than `basic_3d`.

## Numerical validation

```bash
python -m engine examples reference_diffusion --output reference.json
python -m engine diffusion --reference-params reference.json --modes 8 --output runs/diffusion
python -m engine analyze --segment runs/diffusion/segment_000 --reference diffusion_neumann_cosine --reference-params reference.json --field u --output runs/diffusion/errors.csv
```

The error export includes volume-weighted L2, relative L2, Linf, and absolute and
relative mass error for saved frames. The diffusion benchmark is separate from
the cultural scenario. See [numerical validation](docs/VALIDATION.md).

## Commands

| Command | Purpose |
|---|---|
| `examples` | List bundled JSON configs or copy one for editing |
| `run` | Spectral simulation and segmented archive |
| `plot`, `video` | View a run, export frames or animation |
| `genealogy` | Separate source/inheritance trees, CSV and DOT |
| `analyze` | Reference errors or frozen-segment Newton/spectrum analysis |
| `diffusion` | Canonical analytic benchmark |
| `finite-volume` | Independent grid-based backend |
| `plot-archive` | View one NPZ archive, including older formats |
| `integrate` | Integrate an existing equation/provider descriptor |
| `newton`, `spectrum` | Advanced analysis using settings in the corresponding numerical module |
| `validate-births` | Statistical check of the birth clock |

Use `python -m engine <command> --help` for options. Paths are relative to the
current working directory. `examples` refuses to overwrite an existing file.

## Model and provenance

- Config JSON supplies numbers, initial fields, boundaries, solver settings and events.
- `engine/model/cultural_v01.py` defines continuous dynamics.
- `engine/model/event_rules.py` defines the stochastic rate and per-T source rules.
- `engine/model/birth_profiles.py` defines newborn spatial initialization.
- `engine/events/` implements structural events and provenance metadata.

Model edits require a source checkout (`python -m pip install -e .`). The model
name `cultural_v01` remains valid for existing configs. The `engine` import
namespace is retained because archived provider/model descriptors refer to it.

See [provenance and event semantics](docs/PROVENANCE_0_9.md),
[archive contract](docs/ARCHIVE_CONTRACT.md),
[migration from v0.9](docs/MIGRATION.md), and [changes](CHANGELOG.md).

[Function-by-function reference](docs/FUNCTION_REFERENCE.md) covers every function
in the runtime package. Inline comments describe inputs, outputs and invariants.

## Development

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -q
```

Runtime dependencies: NumPy, SciPy, Matplotlib and Pillow. SymPy is needed only
for legacy regression fixtures, installed by the `test` extra. The author-reported local release test run completed **124/124 automated tests**
with `OK` (October 2026). The GitHub CI workflow is configured for Python
3.10 and 3.12 on Linux and Windows; its status must be checked after the first
push. It builds a wheel and smoke-tests the installed wheel from outside the
checkout. Test fixtures are excluded from wheels.

This is a research implementation, not a guarantee of accuracy for arbitrary
parameters. Validate grid/mode convergence for each experiment. Finite spectral
projection can produce ringing and small negative values. Historical P sources
are not reinterpreted as current hosts. Existing archive schema remains 1.0.

## License

Distributed under the [MIT License](LICENSE). Third-party Python dependencies
are subject to their own licenses.

## DOI
10.5281/zenodo.23476661 (https://doi.org/10.5281/zenodo.23476661)
