# Release checks

## Manuscript release: v1.0 (intended tag `v1.0-paper`)

- Author-reported local test run on Windows in October 2026:
  `python -m unittest discover -s tests -q` finished with
  `Ran 124 tests` and `OK`.
- The release source keeps the 1D/2D/3D engine and the same numerical model,
  event mechanisms and run archive schema as the supplied v1.0 source.
- The public `run --help` description uses ASCII `->` in place of a Unicode
  arrow to avoid encoding errors in Windows subprocesses using cp1252.
- The repository adds the MIT license and explicitly identifies the 2D
  manuscript configuration under `paper/representative_2d.json`.
- The config in `paper/representative_2d.json` was compared with the
  `global_config` of the reported run manifest; the JSON objects matched.
- The repository CI checks supported Python 3.10/3.12 on Windows and Linux.
  Check GitHub Actions results after uploading; these jobs are not claimed as
  completed by this local note.

To verify a checkout on a supported Python version:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -q
```

## Historical notes: v0.9.2

Local validation on Linux / Python 3.12:

- 124 unittest tests passed: the 119 existing checks and five saved-time tests.
- Time tests cover nearest/directional selection, earlier-time ties, repeated
  event times, before/after event state, terminal and single-frame cases, invalid
  options, nonfinite/out-of-range requests, CLI PNG export and index compatibility.
- The real event-boundary test reconstructs one pre-birth field and two post-birth
  fields at the same timestamp, preventing silent interpolation across events.
- Wheel installed in a separate virtual environment (shared dependency packages);
  engine imported from that environment's site-packages. Exported 2D and 3D PNGs
  by time from old 0.9.1 archives, without rerunning those simulations.
- 220 runtime functions covered by existing docstrings or 167 new adjacent
  explanatory comments; docs/FUNCTION_REFERENCE.md includes their signatures.
- Executable AST comparison with 0.9.1 shows changes only in the plot CLI,
  spectral viewer and version declaration. PDE, hazard, segment pipeline,
  archive schema, inheritance and event algorithms are unchanged.
- Immutable historical fixtures were not edited.

Comments in cultural_v01.py change the byte-level model-source hash, despite
unchanged formulas. Old-run viewing/reference-error reconstruction still works;
RHS/Newton re-evaluation requires the recorded model source, as before.

CI is configured for Python 3.10/3.12 on Linux and Windows; those remote jobs
have not been run here. Full-sized sandbox runs and MP4 were not repeated for
this viewer/documentation patch.

Reproduce checks:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -q
```
