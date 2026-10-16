# Upgrade 0.9.2 → 1.0

The English user guide is now `USER_GUIDE.md` at the project root. The Russian user guide was removed. Reinstall from the new release directory; `python -m engine --version` should report `1.0`. Configuration and archive formats are unchanged.

# Обновление 0.9.1 → 0.9.2

Распакуйте новую версию в отдельную папку и выполняйте команды из неё:

```powershell
python -m pip install .
python -m engine --version
python -m engine plot --manifest C:/path/to/old/runs/sandbox_2d/manifest.json --time 12.5 --save frame_t12_5.png
```

Замените путь на свой. Старую папку runs можно оставить на месте; повторный расчёт
для просмотра не нужен. Версия должна показывать 0.9.2. Если запускать
`python -m engine` из папки старых исходников, Python может загрузить именно старый
engine, даже после установки нового пакета. Поэтому сначала смените рабочую папку.

Прежние `--frame` и анимация сохранены. Новый `--time` выбирает сохранённый кадр,
а не интегрирует систему до нового времени. Комментарии к модели меняют её хеш;
для Newton/RHS старых run нужен их исходник, для просмотра — нет.

---

# Migration: 0.9 → 0.9.1

This release cleans packaging and command names. It does not change model
formulas, stochastic algorithms, structural events, field IDs, or archive schema.
Config contents and existing runs remain supported. The version is 0.9.1 so runs
can distinguish this package layout from the earlier 0.9.0 source bundle.

## Commands

Install the package, then replace old filename-based commands:

| Old | New |
|---|---|
| `python run_spectral.py` | `spatial-engine run` |
| `python RUN_to_MATPLOT.py` | `spatial-engine plot` |
| `python RUN_to_VIDEO.py` | `spatial-engine video` |
| `python analyze_segment.py` | `spatial-engine analyze` |
| `python analyze_genealogy.py` | `spatial-engine genealogy` |
| `python ARCHIVE_to_MATPLOT.py` | `spatial-engine plot-archive` |
| `python ODE_to_ARCHIVE.py` | `spatial-engine integrate` |
| `python ODE_NEWTON.py` | `spatial-engine newton` |
| `python NEWTON_to_SPECTRUM.py` | `spatial-engine spectrum` |
| `python -m examples.diffusion_reference` | `spatial-engine diffusion` |
| `python -m tools.run_finite_volume` | `spatial-engine finite-volume` |
| `python -m tools.validate_birth_statistics` | `spatial-engine validate-births` |

Existing command arguments are retained. Old wrapper files are intentionally
removed rather than keeping two parallel entrypoint sets. `python -m engine`
is an equivalent entry point. Run `spatial-engine examples` to list bundled
configs, and copy one with `--output`. In a source checkout those nine configs
are in `engine/config/`; additional old configs are private regression fixtures.

## Python imports

| Old module | New module |
|---|---|
| `trajectory_archive` | `engine.archive` |
| `rhs_provider` | `engine.numerics.rhs` |
| `ODE_to_ARCHIVE` | `engine.numerics.ode` |
| `ODE_NEWTON` | `engine.numerics.newton` |
| `NEWTON_to_SPECTRUM` | `engine.numerics.spectrum` |
| `ARCHIVE_to_MATPLOT` | `engine.visualization.archive` |
| `analyze_segment` | `engine.cli.analyze` |
| `examples.diffusion_reference` | `engine.cli.diffusion` |

Function signatures are preserved. `engine.analysis.genealogy.analyze_genealogy`
is unchanged. The private historical archive viewer's `resolve_path` helper now
resolves relative paths from the current working directory, like the CLI.

The engine namespace and `cultural_v01` model identifier stay unchanged for
stored model/provider descriptors. Configs and archives are data-compatible;
external scripts using old Python import names must update the imports above.

## Removed from the public runtime

- Top-level script wrappers and separate generic `examples`, `tools`, `legacy` packages.
- Generated animations, bytecode/cache directories and empty run-output folders.
- Stacked historical READMEs, stale inventories, local timing logs and old release audits.
- The hardcoded dependency lock from another environment; dependency ranges are
  declared once in requirements.txt and installed through pyproject.toml.
- The redundant multi-policy birth demo and local cross-version timing harness.

Three historical files remain unchanged under `tests/fixtures/legacy/` solely to
verify old equation/archive compatibility, with their original hash checks.
They are excluded from the wheel. The source release includes tests and fixtures.
SymPy is now a test extra, not a runtime dependency.

No license or authorship has been invented. The repository owner must select
and add a LICENSE before describing the project as open source. This patch does
not upload to PyPI or create a public repository.
