# Frozen trajectory contract — 1.0

This is a universal mathematical trajectory contract, not a cultural schema.
`engine/archive.py` is the reusable NumPy-only reader/validator/atomic writer.
`read_archive` always uses `allow_pickle=False`; every array must be numeric,
boolean or Unicode/bytes, never object. Floating arrays must be finite.

## Required keys for all supported archives (including legacy)

| Key | Contract |
|---|---|
| `time` | real `(S,)`, nonempty, strictly increasing within one archive |
| `coefficients` | real `(S,D)`; columns are DOFs, not necessarily fields |
| `fields` | string `(D,)`, identifies the field of each column |
| `modes` | integer `(D,dimension)`, nonnegative, unique `(field,mode)` pairs |
| `parameters_json` | scalar JSON text |
| `metadata_json` | scalar JSON object including representation when nonlegacy |

New writers additionally emit `archive_schema_version="1.0"`. Unversioned
archives remain readable. Unsupported major versions fail; additive optional
keys do not force a new schema. Never infer physical field count from D.

## Standard emitted keys, optional for legacy reading

- `initial_coefficients`: `(D,)`, equal to first trajectory row.
- `order_json`: list of `[field, [mode...]]`, exactly agrees with columns.
- `equations_text`: legacy scalar ODE text or JSON callable-provider descriptor.
- `basis_json`: tensor basis, grid, lengths, normalization, requested/effective
  modes, Fourier slot convention; required for tensor_galerkin reconstruction.
- `rhs_provider_json`: equivalent provider descriptor for re-evaluating RHS.
- `field_metadata_json`, `boundary_conditions_json`, `model_metadata_json`.
- `observables_*`: cached diagnostics. Component counts may be stale; paired
  `components_checked_at` times describe when checked, -1 means unavailable.
- `rng_json`: seed/state at segment closure.
- `accessibility`: numeric spatial M when present.
- `events_json`, `genealogy_json`: optional event/registry information. The
  segmented path uses manifest/event_in/out for run-wide genealogy and events.
- For large FV event logs: `events_utf8` uint8 bytes, `events_offsets` integers,
  `events_time` numeric and `events_encoding`; `iter_events` abstracts both forms.

## Representation and reconstruction

`metadata_json.representation` absent means original cosine Galerkin.
`cosine_galerkin`: physical cosine amplitudes; mode k starts at zero, domain
lengths in parameters_json. Pure-cosine new archives preserve this convention.
`tensor_galerkin`: sine k=1..K; Fourier slots 0=constant, 2k−1=cos, 2k=sin.
Tensor product and DOF flattening are C-order. Read basis_json, do not guess.
`cell_average`: historical FV archive; mode indices are cell coordinates;
grid_json defines shape/placement. Registry uses zero padding before field
birth and after retirement. This existing FV representation is retained.

## One spectral archive = one fixed-dimensional system

An archive describes a fixed ORDER, field list and RHS over its interval.
Transient forcing may depend on time but its definition is fixed in that
segment. A state jump, field birth/death/split or operator change closes it.
The closing endpoint is pre-event, next segment starts post-event at the same
physical time. Both archives have internally strictly increasing time.
No variable-length spectral trajectory array or auxiliary hazard DOF is stored.
The hazard clock belongs to event metadata, not to the physical state space.

## Run manifest — 1.0

`schema_version` remains 1.0. `run_manifest_schema_version=1.0` is an additive
explicit alias; `software_version=0.3.0` records the release. Old manifests
without either added key remain readable. Contents: run_id, status, seed,
global_config, RNG identity/state, ordered segments, field_registry, genealogy,
events and relations. Each segment has id, relative path, archive, equations,
start/end time, field_ids, DOF, closure_reasons, projection_in.

A relation records from/to, event time/IDs and the operation
`reconstruct -> event -> project`, with projection error. For all-fields-extinct
or final-time events, terminal_state.npz records the physical post-event state;
no zero-length or zero-dimensional ODE is invented. This is a run-level endpoint,
not a new trajectory schema. Failed runs set status=failed and error.

## Self-contained data vs executable mathematics

Reconstruction/plotting needs archive arrays/basis descriptors only. Arbitrary
Python RHS evaluation requires the matching trusted model module and its
transitive dependencies. The archived descriptor plus source snapshot is an
audit record, not a sandboxed executable bundle. Keep the project release and
requirements-lock.txt with research data. The current source hash covers the
main model file, not its complete import graph. Never execute untrusted model
source merely to view an archive.
