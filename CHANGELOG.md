# Changelog

## 1.0

- Publish the English USER_GUIDE.md at the project root.
- Remove the Russian user guide and update documentation links.
- Update package and CLI version to 1.0; numerical behavior and archive format are unchanged.

## 0.9.2

- `plot --time` selects a saved frame by model time, with nearest/before/after
  policies and explicit event-side selection at repeated times. The CLI reports
  actual time/index/segment and rejects out-of-range requests.
- Every runtime function now has an adjacent explanation or existing docstring;
  167 new function comments and a 220-function reference were added.
- Windows-friendly `python -m engine` examples and expanded user/developer guides.
- PDE, Galerkin pipeline and archive schema unchanged. Model-file comments do
  change its byte hash: old-run RHS/Newton analysis still needs recorded source.

## 0.9.1

Documentation update: Russian user/developer guides and inline extension-point
comments in 18 modules. Executable code and the archived model source are unchanged.

- One installed `spatial-engine` CLI, also available as `python -m engine`.
- Moved archive IO, ODE helpers, analysis and viewers into the engine package;
  removed imports from loose root scripts and standardized module names.
- Nine bundled configs can be copied from any directory with `examples`.
- Isolated historical scripts/configs as regression fixtures; removed generated
  files, duplicate entrypoints and obsolete reports from the distribution.
- Replaced the accumulated README with installation, run and validation guides.
- Added install-resource/CLI tests, wheel packaging and a cross-platform CI workflow.
- Runtime: NumPy, SciPy, Matplotlib, Pillow. SymPy remains a test-only dependency.
- Model equations, numerical algorithms and archive schema 1.0 are unchanged.

See docs/MIGRATION.md for intentional command/import path changes.

## 0.9.0

Per-T birth hazards, separate source/inheritance provenance, hierarchical lineage
labels, semantic event targeting and genealogy exports. See docs/PROVENANCE_0_9.md.
