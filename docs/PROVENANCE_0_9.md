# Spatial Engine v0.9 — provenance and event semantics

This is a limited additive patch of v0.8. The existing 1D/2D/3D grids, PDE model,
Galerkin bases, integration/segment pipeline and trajectory archive schema 1.0
remain in place. No structural event class or physical mechanism was added.

## Three independent relations

- **Spatial/cultural origin:** `source_T_id`, `source_T_lineage`,
  `source_T_lineage_at_birth`, `source_T_display_label`.
- **Parameter inheritance:** `inheritance_parent_id`,
  `inheritance_parent_lineage`, `inheritance_parent_display_label`.
- **Tree position:** `lineage_path`, `lineage_root_id`, `display_label`.

The registry also adds `split_parent_id`, `source_birth_ordinal` and
`birth_event_id`. Initial fields have no creating event, so their event ID is
null. Initial/imported P without a known source has null source metadata.
Existing `parent_id`, `generation` and `genealogy_label` remain compatible legacy
fields. P generation retains its legacy meaning; use the explicit source and
inheritance relations when analyzing provenance.

Machine IDs are not display labels and are never renamed during a run. New
v0.9-mode children use collision-checked `field_000001` style IDs. Existing IDs,
including user-specified IDs, are retained. Tree paths are finite words over
nonnegative integers, not binary strings or p-adic arithmetic.

T root has path `[]` and label `T`. Splitting T_[1] into three components produces
paths `[1,0]`, `[1,1]`, `[1,2]` and labels `T_1.0`, `T_1.1`, `T_1.2`.
Components are ordered by descending component mass, then ascending Cartesian
positive-mass-weighted centroid, then first flat cell index. Ordering precedes
the existing residual-tail assignment and is recorded in the split event.
Multiple initial T roots have separate stable-ID namespaces; subsequent roots
use labels `T@<id>` to avoid collisions.

For P, `lineage_path=[ordinal]` is within the source's namespace. Source lineage
is stored separately. Birth ordinals count all historical births from the same
T ID, including retired P, so indices are never reused. For example:

| Source | Successive labels | P paths | Inheritance |
|---|---|---|---|
| T, path [] | P_root;0, P_root;1, P_root;2 | [0], [1], [2] | independent choice for each |
| T_0, path [0] | P_0;0, P_0;1 | [0], [1] | may cross source branches |
| T_1, path [1] | P_1;0, P_1;1, P_1;2 | [0], [1], [2] | may be null |

These P fields are source siblings. A shared source does not imply parameter
inheritance. P_1;2 can inherit from P_0;0 without changing either source label.
When a T later splits, its P fields keep historical source provenance and remain
active. No automatic P splitting, reassignment or current-host interpretation
is introduced.

## Migration and stochastic source selection

Old configs require no migration. Omitted `source_mode` means `global_T_sum`:
the old global rate law and summed-T birth profile remain active. Uniform birth
also remains unchanged. This legacy mode consumes no additional source RNG draw.
Legacy IDs and component ordering are preserved unless `lineage.enabled=true`.
`source_mode=per_T` automatically enables v0.9 child IDs and split ordering.

Merge these keys into an existing `stochastic_events.p_birth` block, keeping
its parameter/mobility settings:

```json
{
  "enabled": true,
  "source_mode": "per_T",
  "source_rate_function": "default_p_birth_rate_for_source",
  "rate_parameters": {"lambda0": 15.0, "alpha": 0.0},
  "birth_profile": "inside_T",
  "birth_profile_parameters": {
    "amplitude": 0.1, "power": 3.0, "epsilon": 1e-12, "fallback": "fail"
  },
  "parameter_policy": "inherit_mutate",
  "inheritance_parent_policy": "weighted_by_mass",
  "fallback": "random"
}
```

`inheritance_parent_policy` takes precedence over the preserved `parent_policy`
alias. First birth has null inheritance parent and uses the configured
random/default fallback. Source selection is independent of inheritance draws.

For active source j, define M_Tj = sum_cells max(T_j,0) * cell_volume.
The editable example source law is:

    Lambda_j = 0                                      if M_Tj = 0
    Lambda_j = lambda0*M_Tj + alpha*M_P_associated(j)  otherwise
    Lambda_total = sum_j Lambda_j
    Prob(source=j | birth) = Lambda_j / Lambda_total

Associated P mass counts active P whose historical `source_T_id` equals j,
using positive cell values. It is not reassigned to T descendants. Notice that
lambda0 in this example is a mass-proportional coefficient, whereas the legacy
global example uses a constant offset: migration changes the configured model
law deliberately. All rates must be finite and nonnegative.

The existing clock integrates Lambda_total and samples exponential thresholds
as before. At the resulting event, source hazards are evaluated on the current
state, then one source is selected with `rng.choice`. A custom rule can be
registered with `@register_source_rate('name')` in `engine/model/event_rules.py`;
its interface is `(source_index, U, fields, grid, t, params) -> float`, with
aligned field arrays and metadata. Rate functions must be deterministic and
side-effect-free. The old `@register_rate` registry remains available.

For selected j, newborn initialization is:

    P_new(x) = A * [max(T_j(x),0)/(max_x max(T_j(x),0)+epsilon)]^q * M(x).

M is the existing accessibility multiplier (one without landscape). Only that
selected field enters the inside_T formula. A birth still adds a global state
dimension; it is not point nucleation. `uniform` remains explicitly available,
including in per_T mode, but does not localize the profile; use `inside_T` for
source-local births.

The default law gives zero probability to zero-positive-mass T. With no eligible
source the total rate is zero. A forced birth with zero total rate fails clearly.
If a selected inside_T source is below the profile epsilon, per_T mode refuses
uniform fallback, which would violate source-local initialization; use a
suitable epsilon or a custom rate suppressing such sources. Old global fallback
semantics are unchanged. A `source_T_id` override is allowed for manual events
only, must have positive source hazard, and is rejected for integrated-hazard
births. No source-selection RNG draw occurs for that explicit override.

## Auditable events and manifest additions

Every new P event includes child ID/label, source ID/label/path, inheritance
ID/label/path, source and total hazards, source-selection probability, profile
and its parameters, birth time/segment/event ID, resulting parameters, and RNG
selection/parameter audit. Profile audit remains scalar/config metadata: no
spatial arrays are added. Legacy global births have null concrete source and
source probability; unavailable global event hazard audit values may be null.

The manifest retains schema version 1.0 and adds `provenance_schema_version`
(version 1.0 of this additive metadata) and `genealogy_edges` with edge types:

- `T_split`: split parent -> T child;
- `P_source_birth`: T source -> P;
- `P_parameter_inheritance`: parameter parent P -> P.

Edges contain stable IDs, birth time and birth event ID. The legacy `genealogy`
list remains for compatibility. Readers can continue ignoring unknown fields.

## Semantic event targeting

Use one target specification per event. Exact IDs keep their old meaning:

```json
{"targets": ["field_000001", "field_000007"]}
```

Semantic alternatives:

```json
{"target_selector": {"mode": "all_active"}}
{"target_selector": {"mode": "family", "families": ["T"]}}
{"target_selector": {"mode": "family", "families": ["T", "P"]}}
{"target_selector": {
  "mode": "lineage_subtree", "root_field_id": "field_000000",
  "include_root": true, "include_descendants": true
}}
```

Selectors resolve when the event is applied. A subtree follows T split and P
source edges, never parameter-inheritance edges. Its root can be inactive; only
active reached fields are targeted. The event records `requested_selector`,
`resolved_target_ids`, and `resolved_target_labels`. Empty matches are a no-op.
Mixing a selector with explicit targets or legacy target_families fails clearly.

For finite-duration semantic events, the resolved IDs are a snapshot at event
start: new children are not silently added during the duration. This policy is
recorded as `snapshot_at_event_start`. Old explicit-ID duration semantics retain
their existing ancestor handling. Selectors work with existing catastrophe,
sabotage, manual, mutation and extinction handlers; no event type is added.

## Run and analyze

```bash
python -m pip install -e .
spatial-engine run --config engine/config/provenance_demo_1d.json --output runs/provenance_09
spatial-engine genealogy --manifest runs/provenance_09/manifest.json \
  --csv runs/provenance_09/genealogy.csv \
  --graphviz runs/provenance_09/genealogy.dot \
  --output runs/provenance_09/genealogy.txt
```

The short deterministic-seed demo starts with one T blob and no P. The validated
run makes five P source siblings, splits T into two, then creates five P from
T_0 and two from T_1: 12 births and 15 segments. The first three are P_root;0,
P_root;1 and P_root;2. The analyzer generates text/CSV/DOT exports from the run.
Inheritance crosses source branches and is printed as a separate forest.

The larger existing 3D scenario has an opt-in variant:

```bash
spatial-engine run --config engine/config/sandbox_3d_per_T.json --output runs/per_T_3d
spatial-engine plot --manifest runs/per_T_3d
```

The analyzer accepts old manifests, warns about unavailable provenance and
leaves unknown source relations unset. It never guesses a T source from names.
It may use an explicitly stored old parent_id plus family to recover old T split
or P inheritance edges. CSV contains field_id, display_label, family,
lineage_path, source_T_id/label, inheritance_parent_id/label, birth_time,
death_time, birth_segment, generation, plus event ID and ordinal. Paths are JSON
arrays in CSV. DOT styles: solid split, dashed source birth, dotted inheritance.
Graphviz is optional; exporting DOT adds no dependency.

## Separation of concerns and numerical validation

- `event_rules.py`: WHEN a new dimension appears and WHICH source is selected.
- `birth_profile`: WHERE its initial spatial profile lies.
- Parameter/mobility policy: FROM WHICH P it inherits parameters.
- `cultural_v01.py`: continuous dynamics afterward.
- `spatial-engine genealogy`: provenance and semantic event audit.
- `spatial-engine analyze` and reference tooling: numerical error, independent of
  the cultural scenario.

The v0.8 diffusion benchmark and volume-weighted L2, relative L2, Linf and mass
metrics remain available, for example:

```bash
spatial-engine diffusion --reference-params engine/config/reference_diffusion_3d.json --modes 4 4 4 --grid 12 14 16 --lengths 1 1.2 1.4 --output runs/diffusion_3d
spatial-engine analyze --segment runs/diffusion_3d/segment_000 --reference diffusion_neumann_cosine --reference-params engine/config/reference_diffusion_3d.json --field u --output runs/diffusion_3d/errors.csv
```

## Limitations

Birth localization is exact on the event grid before the unchanged Galerkin
projection. A finite basis can produce ringing/leakage; this patch does not
impose positivity or alter projection. Periodic split centroids use Cartesian
coordinates for deterministic ordering, not circular geometry. No current host
is inferred after T changes. Old archives cannot recover source provenance they
never stored. Simultaneous events follow the existing ordering; source hazards
are recomputed on the state at actual birth application. A forced birth after
all eligible sources have disappeared fails rather than inventing a source.

The full-sized 3D sandbox is a supplied example; regression testing uses small
3D runs. No p-adic arithmetic, binary branching restriction, new PDE, adaptive
basis, new structural events or GUI is included.
