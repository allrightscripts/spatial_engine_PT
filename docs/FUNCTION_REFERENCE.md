# Справочник функций

Версия 1.0. Ниже перечислены функции рабочего пакета engine, включая методы
и вложенные callbacks. Это навигация по коду, а не обещание стабильности частного API.
Для практических примеров начните с ../USER_GUIDE.md и EXTENDING.md.

Комментарии и docstrings оставлены на английском в стиле исходников.

## engine/analysis/diagnostics.py

[Открыть исходник](../engine/analysis/diagnostics.py)

### `observe`

```python
observe(sim, t)
```

Compute masses, family counts, entropy and functional value from current state. Component counts reuse the detector cache and include their last-check times.

## engine/analysis/error_metrics.py

[Открыть исходник](../engine/analysis/error_metrics.py)

### `error_metrics`

```python
error_metrics(u_num, u_ref, grid, epsilon=1e-12)
```

Compare equal-shaped real samples using cell-volume-weighted L2 and mass, plus Linf and regularized relative errors. Epsilon affects denominators, not the difference field.

## engine/analysis/genealogy.py

[Открыть исходник](../engine/analysis/genealogy.py)

### `load_genealogy`

```python
load_genealogy(path)
```

Load registry metadata and derive typed source/inheritance edges. Reject duplicate IDs or missing parents; legacy unknown sources stay unknown.

### `_forest`

```python
_forest(registry, edges, allowed, families=None)
```

Format only the allowed edge types as an indented forest, checking multiple parents/cycles. Source and inheritance forests are deliberately separate calls.

### `_forest.key`

```python
_forest.key(i)
```

Sort siblings deterministically by birth time, lineage path and stable ID; labels never define identity.

### `_forest.visit`

```python
_forest.visit(i, indent, stack)
```

Depth-first traversal appends readable labels and inactive markers; stack/visited sets detect invalid cycles and duplicate visits.

### `analyze_genealogy`

```python
analyze_genealogy(manifest, csv_path=None, graphviz=None)
```

Build two forests and table rows, optionally export CSV/DOT, and return text/rows/edges/warnings. Reads metadata only, never reruns dynamics.

## engine/analysis/reference_errors.py

[Открыть исходник](../engine/analysis/reference_errors.py)

### `analyze_reference`

```python
analyze_reference(segment, reference, params=None, fields=None, epsilon=1e-12, output=None)
```

Compare every saved time on the archive grid; never evaluate model RHS.

One reference parameter dictionary is used for every selected field.
Select one field when references differ between fields.

## engine/analysis/reference_solutions.py

[Открыть исходник](../engine/analysis/reference_solutions.py)

### `register_reference`

```python
register_reference(name)
```

Return a decorator exposing a callable under a JSON-selectable reference solution name. Register rules at module import, not during integration.

### `register_reference.register`

```python
register_reference.register(function)
```

Store the supplied reference solution callable under the captured name and return it unchanged; registration itself does not execute the rule.

### `_finite`

```python
_finite(value, name)
```

Convert a reference parameter to float and reject NaN/Inf with its parameter name in the error.

### `diffusion_neumann_cosine`

```python
diffusion_neumann_cosine(grid, t, params)
```

Evaluate one cosine product or a sum of products on grid.mesh, with analytic diffusion decay and optional constant offset. Validate axis counts and nonnegative integer wave numbers.

### `reference_solution`

```python
reference_solution(name, grid, t, params=None)
```

Dispatch by registered reference name and return validated real grid samples for the given time/parameters.

## engine/archive.py

[Открыть исходник](../engine/archive.py)

### `json_array`

```python
json_array(value)
```

Portable scalar Unicode JSON; reject NaN/Infinity and Python objects.

### `decode`

```python
decode(data, key, default=None)
```

Decode a scalar JSON array from an archive dictionary; use default when the key is absent. Does not evaluate Python code.

### `validate`

```python
validate(data)
```

Validate legacy and versioned payloads without executing model code.

### `write_archive`

```python
write_archive(path, data)
```

Validate then atomically replace one self-contained NPZ (no pickle).

### `read_archive`

```python
read_archive(path)
```

Load NPZ with allow_pickle=False, copy arrays out of the file, then validate the archive contract. Returns the array dictionary, not reconstructed fields.

### `reconstruct_grid`

```python
reconstruct_grid(data, index)
```

Exact archived cell averages, shape (registered fields, *grid.shape).

### `pack_events`

```python
pack_events(events, json_limit=1000)
```

Small logs stay JSON; large logs use indexed UTF-8 records in numeric arrays.

Offsets permit per-event decoding without parsing one giant JSON document.
NPZ compression keeps a single portable file; no external sidecars required.

### `iter_events`

```python
iter_events(data)
```

Yield decoded event records from packed archive metadata; event order is preserved for replay/audit.

## engine/cli/__init__.py

[Открыть исходник](../engine/cli/__init__.py)

### `export_example`

```python
export_example(argv)
```

List bundled JSON examples or copy one to a new editable file. Exclusive creation protects an existing experiment from overwrite.

### `main`

```python
main(argv=None)
```

Dispatch one installed subcommand lazily, forwarding its arguments and restoring sys.argv afterward. This entrypoint is shared by python -m engine.

## engine/cli/analyze.py

[Открыть исходник](../engine/cli/analyze.py)

### `analyze`

```python
analyze(folder, time=None, start='last', output=None)
```

Analyze one frozen RHS; output defaults to the selected segment for API compatibility.

## engine/cli/diffusion.py

[Открыть исходник](../engine/cli/diffusion.py)

### `run_benchmark`

```python
run_benchmark(destination, params, modes=(8,), shape=(128,), lengths=(2.0,), t_end=0.2)
```

Build a pure-diffusion run and initialize physical samples from the exact solution before projection. Do not reproduce this by running its saved config alone.

### `main`

```python
main()
```

Parse reference parameters, axes and resolution, run the canonical diffusion benchmark and report its segment count.

## engine/cli/finite_volume.py

[Открыть исходник](../engine/cli/finite_volume.py)

### `main`

```python
main()
```

Run the independent FV solver from JSON, write a standard NPZ archive and reload it to report its shape/genealogy.

## engine/cli/genealogy.py

[Открыть исходник](../engine/cli/genealogy.py)

### `main`

```python
main()
```

Parse manifest/export destinations, print source and inheritance forests separately, then optionally write text, CSV and DOT outputs.

## engine/cli/plot.py

[Открыть исходник](../engine/cli/plot.py)

### `main`

```python
main(argv=None)
```

Parse viewer options, render the archive, and report the actual saved time.

argv is optional for embedding/tests. User mistakes become concise argparse
errors. Numerical model time has the units chosen by the model, not playback
seconds; --fps only affects animations. --time does not interpolate frames.

## engine/cli/run.py

[Открыть исходник](../engine/cli/run.py)

### `main`

```python
main()
```

Read config JSON, launch a segmented simulation and print completion status. Expected input errors use argparse; --debug retains tracebacks.

## engine/cli/validate_births.py

[Открыть исходник](../engine/cli/validate_births.py)

### `validate`

```python
validate(count=4096, rate=2.0)
```

Sample constant-rate clock waiting times and return statistical diagnostics; this validates stochastic timing independently of spatial dynamics.

## engine/cli/video.py

[Открыть исходник](../engine/cli/video.py)

### `main`

```python
main()
```

Validate output/format options and export saved frames to GIF or MP4. FFmpeg is needed for MP4; no model integration runs here.

## engine/events/base.py

[Открыть исходник](../engine/events/base.py)

### `Event.record`

```python
Event.record(self)
```

Convert the event dataclass to a serializable dictionary and add its structural event_type tag.

### `from_config`

```python
from_config(data)
```

Build the existing event class selected by type, moving a top-level semantic selector into parameters. Reject unknown event types.

## engine/events/birth_source.py

[Открыть исходник](../engine/events/birth_source.py)

### `select_source`

```python
select_source(sim, config, event)
```

Return selected T registry index plus scalar hazard/probability audit. Legacy global mode returns no concrete source and consumes no source-selection draw.

## engine/events/hazard.py

[Открыть исходник](../engine/events/hazard.py)

### `BirthClock.__init__`

```python
BirthClock.__init__(self, config, rng, seed)
```

Initialize integrated hazard, birth count and RNG state; draw an exponential threshold only if the clock is enabled.

### `BirthClock.enabled`

```python
BirthClock.enabled(self)
```

Check the enable flag and optional max_births limit without consuming RNG or changing clock state.

### `BirthClock.renew`

```python
BirthClock.renew(self)
```

Reset accumulated hazard and draw a positive Exp(1) threshold, recording the generator state before the draw.

### `BirthClock.integral_clock`

```python
BirthClock.integral_clock(self, provider)
```

Generic integration hook: auxiliary scalar is never a physical DOF.

### `BirthClock.integral_clock.rate`

```python
BirthClock.integral_clock.rate(t, a)
```

Reconstruct the trial physical state and evaluate total birth intensity. Must remain deterministic across adaptive solver retries.

### `BirthClock.audit`

```python
BirthClock.audit(self)
```

Return the current threshold, integrated hazard, birth number and RNG provenance as event metadata.

### `BirthClock.consumed`

```python
BirthClock.consumed(self)
```

Count a committed stochastic birth and renew the threshold only if further births remain enabled.

## engine/events/landscape.py

[Открыть исходник](../engine/events/landscape.py)

### `landscape`

```python
landscape(sim, event)
```

Install/remove an accessibility layer and recompute the product of active layers. Alters transport/reaction accessibility, not existing field values.

## engine/events/operators.py

[Открыть исходник](../engine/events/operators.py)

### `spatial_mask`

```python
spatial_mask(grid, geometry, t=0.0)
```

Backward-compatible entry point into the shared Python mask registry.

### `suppress`

```python
suppress(sim, event)
```

Impulse suppression, optionally held until time+duration.

A held mask multiplies by (1-strength*mask) after each integration substep;
partial held suppression is step-dependent, intentionally not a rate law.
Use duration=0 for a time-step independent impulse.

### `nucleate`

```python
nucleate(sim, event)
```

Create a field from amplitude times a geometric mask using the existing nucleation event. Set event.targets to the newborn ID; distinct from per_T P birth.

### `mutate`

```python
mutate(sim, event)
```

Explicit parameter replacement; stochastic perturbation happens upstream.

### `extinct`

```python
extinct(sim, event)
```

Retire each active resolved target at event time, preserving its registry row for historical analysis.

### `manual`

```python
manual(sim, event)
```

Minimal intervention: add a nonnegative spatial field increment.

### `PoissonNucleation.__init__`

```python
PoissonNucleation.__init__(self, config)
```

Store configuration for the legacy per-step Poisson nucleation mechanism; it is independent of the integrated P-birth clock.

### `PoissonNucleation.step`

```python
PoissonNucleation.step(self, sim, dt, t)
```

Draw a Poisson count over dt, then random centres, using sim.rng only; apply and audit each generated nucleation event.

## engine/events/p_birth.py

[Открыть исходник](../engine/events/p_birth.py)

### `bounds`

```python
bounds(pair)
```

Validate a finite ordered pair [low, high] and return float endpoints; used by parameter and mobility policies.

### `mutate_value`

```python
mutate_value(value, sigma, rng)
```

Apply additive Gaussian mutation using the supplied generator. Sigma=0 returns the value without consuming a random draw.

### `select_parent`

```python
select_parent(sim, config)
```

Return the registry index of an active P inheritance parent, or None if none exists. Explicit parent_id overrides selection; zero total mass falls back to uniform selection.

### `create_parameters`

```python
create_parameters(sim, c, parent)
```

Build scalar parameters by manual, random or inherit/mutate policy and explicit fallback; clip configured ranges and return values plus audit metadata.

### `expand_mobility`

```python
expand_mobility(sim, c, fields, parent, parameters, t)
```

Prepare an enlarged field mobility matrix while preserving its old block; validate symmetry/PSD when requested. Return matrix and audit before appending the child.

### `p_birth`

```python
p_birth(sim, event)
```

Select spatial source and construct its profile, then independently generate inherited/random parameters and mobility. Append one global field and enrich event metadata; no integration occurs here.

## engine/events/provenance.py

[Открыть исходник](../engine/events/provenance.py)

### `enabled`

```python
enabled(sim)
```

Enable v0.9 naming/order when lineage.enabled or per_T birth is configured; old configs retain legacy generated IDs/order.

### `next_field_id`

```python
next_field_id(sim)
```

Find an unused stable field_NNNNNN ID without deriving it from a display label. Historical retired IDs remain reserved.

### `initialize`

```python
initialize(sim, child, parent=None)
```

Add provenance defaults to a newly appended registry row; root T labels use distinct namespaces when multiple roots exist.

### `set_split`

```python
set_split(child, parent, branch)
```

Set the explicit split parent, inherited root namespace and parent path plus child index. Updates metadata only, not field values.

### `set_P`

```python
set_P(sim, child, source, inheritance)
```

Assign a per-source historical birth ordinal, source lineage and separate inheritance relation. Retired siblings still count, so ordinals are not reused.

### `edges`

```python
edges(registry)
```

Derive typed T_split, P_source_birth and P_parameter_inheritance edges from explicit IDs, including creation time/event ID.

### `record_creation`

```python
record_creation(sim, event)
```

Attach the recording event ID to its newborn fields. Initial fields have no creating event and retain null IDs.

## engine/events/splitting.py

[Открыть исходник](../engine/events/splitting.py)

### `components`

```python
components(u, threshold, periodic_axes)
```

Face connectivity (2 neighbours in 1D, 4 in 2D, 6 in 3D), including torus seams.

### `components.root`

```python
components.root(a)
```

Find and compress the union-find representative joining periodic boundary labels; this is component identity, not genealogy.

### `SplitDetector.__init__`

```python
SplitDetector.__init__(self, config)
```

Validate cadence/threshold settings and initialize component-history caches; these caches track isolation duration across checks.

### `SplitDetector.check`

```python
SplitDetector.check(self, sim, t)
```

Find sufficiently large connected components in eligible active fields. Preserve isolation age only for unambiguous component correspondence, then propose/apply a split.

### `SplitDetector.split`

```python
SplitDetector.split(self, sim, index, masks, t)
```

Partition without destroying mass; residual tails belong to largest lobe.

Every cell keeps its location and value. Parent retires, daughters append
to the global registry. This prevents any historic DOF renumbering.

### `SplitDetector.split.key`

```python
SplitDetector.split.key(mask)
```

Order components by descending mass, Cartesian centroid and first flat cell index. Evaluate before residual tails are assigned to the largest component.

## engine/events/targeting.py

[Открыть исходник](../engine/events/targeting.py)

### `resolve_selector`

```python
resolve_selector(registry, selector)
```

Return active registry rows matching all/family/source-subtree semantics. Subtrees traverse split/source links, never P parameter inheritance.

### `resolve_event`

```python
resolve_event(sim, event)
```

Resolve semantic targets at application time and store their stable IDs/labels. Duration-end events keep start-time resolution; mixed selector/exact targets fail.

## engine/galerkin/basis.py

[Открыть исходник](../engine/galerkin/basis.py)

### `apply_axis`

```python
apply_axis(array, matrix, axis)
```

Apply matrix (output,input) along one tensor axis, retaining axis order.

### `TensorBasis.__init__`

```python
TensorBasis.__init__(self, grid, kinds, modes, dealiasing='none')
```

Validate per-axis basis/mode choices and precompute separable transforms, derivatives and normalization. Fourier requests are frequency cutoffs, while sine/cosine requests count basis functions.

### `TensorBasis.from_config`

```python
TensorBasis.from_config(cls, grid, fields, settings)
```

Auto basis from common homogeneous BCs. No silent boundary lifting.

### `TensorBasis.project`

```python
TensorBasis.project(self, values)
```

Tensor projection of trailing spatial axes, preserving leading batches.

### `TensorBasis.reconstruct`

```python
TensorBasis.reconstruct(self, coefficients)
```

Transform trailing mode axes into physical-grid axes, retaining leading field/batch axes. Coefficients use this basis' amplitude normalization.

### `TensorBasis.derivative`

```python
TensorBasis.derivative(self, coefficients, axis)
```

Exact derivative of the represented trigonometric polynomial at grid nodes.

### `TensorBasis.gradient`

```python
TensorBasis.gradient(self, coefficients)
```

Reconstruct spatial derivatives along each physical axis using derivative matrices; coordinates and lengths determine their units.

### `TensorBasis.laplacian`

```python
TensorBasis.laplacian(self, coefficients)
```

Coefficient-space Laplacian; reconstruct its output for spatial values.

### `TensorBasis.weak_divergence`

```python
TensorBasis.weak_divergence(self, fluxes)
```

Project div(flux) = -<grad(phi),flux>/||phi||² (zero boundary term).

Periodic cancellation, Dirichlet test functions zero, or natural zero
Neumann flux are required. This avoids imposing an incorrect parity on
fluxes when sine/cosine potentials are differentiated.

### `TensorBasis.descriptor`

```python
TensorBasis.descriptor(self)
```

Serialize grid, basis and truncation settings as ordinary data so archives can reconstruct the same basis.

### `TensorBasis.from_descriptor`

```python
TensorBasis.from_descriptor(cls, descriptor)
```

Rebuild the Grid and TensorBasis from stored settings; no model reaction code is needed for reconstruction.

## engine/galerkin/integration.py

[Открыть исходник](../engine/galerkin/integration.py)

### `IntegrationSession.__init__`

```python
IntegrationSession.__init__(self)
```

Start without a live solver. One session belongs to one unchanged segment; never reuse it after a structural event.

### `IntegrationSession.solve`

```python
IntegrationSession.solve(self, rhs, interval, initial, *, method, t_eval, rtol, atol, max_step, events=None)
```

solve_ivp-shaped result; optional upward terminal integral-clock root.

## engine/galerkin/provider.py

[Открыть исходник](../engine/galerkin/provider.py)

### `GalerkinProvider.__init__`

```python
GalerkinProvider.__init__(self, config, fields, basis, forcing=None, accessibility=None)
```

Freeze active fields, basis, forcing and accessibility for one fixed-dimensional ODE. Build stable coefficient ordering and validate compatible mobility.

### `GalerkinProvider.physical_state`

```python
GalerkinProvider.physical_state(self, state)
```

Reuse reconstruction between RHS and deterministic hazard evaluation.

The last coefficient array is copied because solve_ivp may reuse its
buffers. Only this most recent evaluation is cached, never a trajectory.

### `GalerkinProvider.rhs`

```python
GalerkinProvider.rhs(self, t, state)
```

Project chemical potential and reaction; weak divergence of L grad(mu).

All work stays within this segment's fixed field/mode ordering. No random
draws or detector updates occur inside an adaptive solver trial RHS.

### `GalerkinProvider.descriptor`

```python
GalerkinProvider.descriptor(self)
```

Describe the frozen RHS as JSON-compatible data, including a byte hash of the model source. Even model comments affect that audit hash.

### `GalerkinProvider.source`

```python
GalerkinProvider.source(self)
```

Return cached JSON text of the RHS descriptor; this is archive data, not executable generated Python.

### `GalerkinProvider.archive_extras`

```python
GalerkinProvider.archive_extras(self)
```

Return cached basis/field/BC/provider metadata arrays for the archive writer, including accessibility when present.

### `from_descriptor`

```python
from_descriptor(d)
```

Restore a provider and verify its model-source hash. A mismatch requires the recorded source, not silently substituting today's model.

## engine/galerkin/run.py

[Открыть исходник](../engine/galerkin/run.py)

### `dump_json`

```python
dump_json(path, value)
```

Write finite JSON via a temporary sibling file and atomic replacement. Parent directories are created; NaN/Inf are rejected.

### `normalize_event`

```python
normalize_event(raw, dimension)
```

Accept explicit v0.2 catastrophe controls and legacy geometry/parameters.

### `PlanningDetector.__init__`

```python
PlanningDetector.__init__(self, config)
```

Initialize component tracking and an empty proposal list; planning must not yet change the live field dimension.

### `PlanningDetector.split`

```python
PlanningDetector.split(self, sim, index, masks, t)
```

Queue the detected component masks for the next boundary commit, avoiding duplicate component detection.

### `EventState.__init__`

```python
EventState.__init__(self, config)
```

Reuse registry/event machinery without FV time stepping; configure accessibility, rate events and the split planner.

### `EventState.apply`

```python
EventState.apply(self, event)
```

During planning, queue events; otherwise expand compatible family targets and delegate handler application/recording. Conflicting target specifications fail.

### `EventState.controlled`

```python
EventState.controlled(sim, event)
```

Apply spectral impulse suppression or install/remove a finite-duration sink. Unlike held FV suppression, a rate sink enters the continuous RHS.

### `EventState.active_forcing`

```python
EventState.active_forcing(self)
```

Resolve active descendants of the original target roots once per segment.

### `SegmentedRun.__init__`

```python
SegmentedRun.__init__(self, config, destination)
```

Validate run settings, initialize event state/clock, and create the output manifest and model-source snapshot. The destination must be empty.

### `SegmentedRun.refresh_manifest`

```python
SegmentedRun.refresh_manifest(self)
```

Refresh registry, typed genealogy edges, events and clock audit from live state, then atomically save manifest.json.

### `SegmentedRun.build_segment`

```python
SegmentedRun.build_segment(self, t)
```

Explicit physical→coefficient map, once after every event batch.

### `SegmentedRun.advance`

```python
SegmentedRun.advance(self, t, target)
```

Integrate a control slab of the SAME ODE; no re-projection/re-solving.

### `SegmentedRun.close_segment`

```python
SegmentedRun.close_segment(self, t, reasons, event_out)
```

Write one immutable mathematical system and all its accumulated samples.

### `SegmentedRun.boundary_proposals`

```python
SegmentedRun.boundary_proposals(self, t, check_split, dt)
```

Plan all mutations without altering the saved pre-event coefficients.

### `SegmentedRun.commit`

```python
SegmentedRun.commit(self, t, events, splits)
```

Event map acts in physical space; next build_segment applies projection.

### `SegmentedRun.run`

```python
SegmentedRun.run(self)
```

Advance to control/event boundaries, close changed segments and commit event maps. Return the final manifest; record failed status before re-raising errors.

## engine/model/birth_profiles.py

[Открыть исходник](../engine/model/birth_profiles.py)

### `register_birth_profile`

```python
register_birth_profile(name)
```

Return a decorator exposing a callable under a JSON-selectable newborn profile name. Register rules at module import, not during integration.

### `register_birth_profile.register`

```python
register_birth_profile.register(function)
```

Store the supplied newborn profile callable under the captured name and return it unchanged; registration itself does not execute the rule.

### `total_T`

```python
total_T(U, fields, grid)
```

Sum active T fields among the supplied aligned arrays/metadata. In per_T mode the caller supplies only the selected T.

### `_amplitude`

```python
_amplitude(params)
```

Convert amplitude to float and require a finite nonnegative value before creating a newborn profile.

### `uniform`

```python
uniform(U, fields, grid, t, params)
```

Return a constant profile plus effective parameter audit; the shared wrapper applies accessibility afterward.

### `inside_T`

```python
inside_T(U, fields, grid, t, params)
```

Normalize clipped T by its maximum plus epsilon and raise to power, handling the configured zero-T fallback. Does not mutate T or apply the landscape mask.

### `evaluate_birth_profile`

```python
evaluate_birth_profile(U, fields, grid, t, config, accessibility=None)
```

Return physical newborn and scalar/config audit; never consume RNG.

## engine/model/cultural_v01.py

[Открыть исходник](../engine/model/cultural_v01.py)

### `CulturalFunctional.chemical_potential`

```python
CulturalFunctional.chemical_potential(self, U, fields, grid, t)
```

Quadratic free-energy derivative mu=U; returned array shares storage with U and must be treated as read-only.

### `CulturalFunctional.value`

```python
CulturalFunctional.value(self, U, fields, grid, t)
```

Integrate one half of sum_i U_i squared using grid.volume, the volume of one cell. Return a scalar diagnostic.

### `family_sum`

```python
family_sum(U, fields, family)
```

Sum all current fields belonging to one family; shape == one field.

### `other_family_members`

```python
other_family_members(U, fields, i, family)
```

Sum a family while excluding U[i] when U[i] belongs to that family.

### `smooth_switch`

```python
smooth_switch(x, threshold, width)
```

Stable logistic switch from 0 to 1 around `threshold`.

### `logistic_reaction`

```python
logistic_reaction(i, U, fields, grid, t, total)
```

Legacy shared logistic law kept for model_type='default'.

### `tradition_v1`

```python
tradition_v1(i, U, fields, grid, t, total)
```

Slow self-limited T dynamics with competition between T lineages.

R_Ti = r_i T_i (1 - T_i/K_i)
       - c_T,i T_i T_other
       - d_i T_i

T does not need to receive a direct energetic benefit from suppressing P.
Its 'despotic' effect is represented in progressive_v1 as a negative term
acting on P.

### `progressive_v1`

```python
progressive_v1(i, U, fields, grid, t, total)
```

P dynamics: self-growth + P-to-P acceleration + T suppression.

Let
    P_other = sum of all other active P fields,
    T_total = sum of all active T fields,
    S(P_total) = logistic threshold switch.

Then
    R_Pi = r_i P_i (1 - P_i/K_i)
           + a_i P_i P_other
           - b_i T_total P_i S(P_total)
           - d_i P_i.

Interpretation used by the demo model:
  * progress_acceleration > 0: existing novelty accelerates other novelty;
  * T_suppression > 0: active T suppresses P;
  * the sigmoid makes suppression strongest after aggregate P becomes
    sufficiently visible/large.

This reaction law is independent from stochastic P-birth hazard: the hazard
controls WHEN a new dimension appears; this law controls what happens to an
already-existing P field afterward.

### `CulturalReaction.validate_type`

```python
CulturalReaction.validate_type(model_type)
```

Require model_type to name an implemented REACTION_LAWS entry; family labels alone do not select a reaction formula.

### `CulturalReaction.__call__`

```python
CulturalReaction.__call__(self, U, fields, grid, t)
```

Dispatch each current field to its model_type reaction and return an array shaped like U. Use aligned field metadata rather than permanent numeric indices.

### `build_model`

```python
build_model(parameters=None)
```

Factory expected by the universal loader.

## engine/model/definitions.py

[Открыть исходник](../engine/model/definitions.py)

### `QuadraticFunctional.chemical_potential`

```python
QuadraticFunctional.chemical_potential(self, u, fields, grid, t)
```

Return mu=U, the variational derivative of the quadratic functional. The returned array aliases the input: downstream code must not mutate it.

### `QuadraticFunctional.value`

```python
QuadraticFunctional.value(self, u, fields, grid, t)
```

Compute one half of the sum of squared field values times cell volume, returning a scalar functional value.

### `DiagonalMobility.__call__`

```python
DiagonalMobility.__call__(self, u, fields, grid, t)
```

Construct a diagonal field mobility tensor from each field's diffusion parameter; singleton spatial axes enable broadcasting.

### `LogisticCompetition.__call__`

```python
LogisticCompetition.__call__(self, u, fields, grid, t)
```

Evaluate local self-limited growth, cross-field competition and decay for every field; return an array with the same shape as u.

### `Model.__init__`

```python
Model.__init__(self, functional=None, mobility=None, reaction=None)
```

Compose functional, mobility and reaction components, using demo defaults for omitted components. Custom components must follow their array contracts.

### `Model.rhs`

```python
Model.rhs(self, u, fields, grid, t)
```

Combine transport of chemical potentials with local reactions on the grid. This generic FV path does not contain cultural family rules.

## engine/model/event_rules.py

[Открыть исходник](../engine/model/event_rules.py)

### `register_rate`

```python
register_rate(name)
```

Return a decorator exposing a callable under a JSON-selectable legacy global rate name. Register rules at module import, not during integration.

### `register_rate.register`

```python
register_rate.register(function)
```

Store the supplied legacy global rate callable under the captured name and return it unchanged; registration itself does not execute the rule.

### `p_birth_rate`

```python
p_birth_rate(U, fields, grid, t, params)
```

EXAMPLE only: lambda0 + alpha * integral(total P). Replace freely.

Family metadata, not array position, identifies P. Spectral roundoff may
produce tiny negative masses; this example counts nonnegative total mass.

### `evaluate_rate`

```python
evaluate_rate(name, U, fields, grid, t, params)
```

Evaluate a named legacy global intensity and reject negative/nonfinite values. Units are births per model-time unit, not probability per step.

### `register_source_rate`

```python
register_source_rate(name)
```

Return a decorator exposing a callable under a JSON-selectable per-T source rate name. Register rules at module import, not during integration.

### `register_source_rate.register`

```python
register_source_rate.register(function)
```

Store the supplied per-T source rate callable under the captured name and return it unchanged; registration itself does not execute the rule.

### `p_birth_rate_for_source`

```python
p_birth_rate_for_source(source_index, U, fields, grid, t, params)
```

Example: lambda0 * positive T mass + alpha * associated positive P mass.

A zero-mass T is ineligible. U and metadata are aligned, including when
called from a fixed-dimensional provider. This is user-editable model code.

### `source_hazards`

```python
source_hazards(config, U, fields, grid, t)
```

Evaluate the registered per-source law for all active T fields; return aligned indices and a finite nonnegative rate array.

### `evaluate_birth_rate`

```python
evaluate_birth_rate(config, U, fields, grid, t)
```

Dispatch between legacy global_T_sum and the sum of per_T hazards. This controls the clock, not newborn spatial initialization.

## engine/model/loading.py

[Открыть исходник](../engine/model/loading.py)

### `load_model`

```python
load_model(config)
```

Import a simple trusted module name under engine.model and invoke build_model(model_parameters). Unknown names produce an explicit model-path error.

## engine/model/masks.py

[Открыть исходник](../engine/model/masks.py)

### `register_mask`

```python
register_mask(name)
```

Return a decorator exposing a callable under a JSON-selectable geometry mask name. Register rules at module import, not during integration.

### `register_mask.register`

```python
register_mask.register(function)
```

Store the supplied geometry mask callable under the captured name and return it unchanged; registration itself does not execute the rule.

### `my_catastrophe`

```python
my_catastrophe(X, Y=None, Z=None, *, t, params)
```

Example custom interval/ellipse/ellipsoid, coefficients supplied by JSON.

### `_builtin_mask`

```python
_builtin_mask(grid, geometry)
```

Masks sampled at cell centres; supports smooth and hard local geometry.

### `evaluate_mask`

```python
evaluate_mask(grid, geometry, t=0.0)
```

Evaluate a named custom mask or built-in geometry at time t, broadcast to grid.shape and require finite values in [0,1].

## engine/model/mobility.py

[Открыть исходник](../engine/model/mobility.py)

### `validate_onsager`

```python
validate_onsager(matrix, dissipative=True, tolerance=1e-10)
```

Validate a finite square matrix; optionally require symmetry and positive semidefiniteness within tolerance. Never silently project it onto a different matrix.

### `ConstantFieldMobility.birth_matrix`

```python
ConstantFieldMobility.birth_matrix(self, U, fields, grid, t)
```

Build the constant field-indexed matrix from diagonal diffusion and stable-ID mobility_row entries; return an n by n array.

### `ConstantFieldMobility.__call__`

```python
ConstantFieldMobility.__call__(self, U, fields, grid, t)
```

Add singleton spatial axes to the constant matrix so transport can broadcast it over the physical grid.

### `ConstantFieldMobility.assign_birth_matrix`

```python
ConstantFieldMobility.assign_birth_matrix(self, fields, matrix)
```

Write the expanded matrix back into field diffusion parameters and ID-keyed coupling rows; mutates the supplied metadata.

## engine/numerics/newton.py

[Открыть исходник](../engine/numerics/newton.py)

### `_check_expression`

```python
_check_expression(tree: ast.AST, allowed_names: set[str], size: int)
```

Inspect the expression AST against the allowed syntax before compilation; this is the restricted legacy equation interface, not arbitrary Python execution.

### `load_equations`

```python
load_equations(source: str)
```

Read ORDER/default parameters and RHS expressions, or detect a provider descriptor. Validate and prepare the representation used by the numerical caller.

### `label`

```python
label(order, i: int)
```

Format a stored field/mode entry into a readable coefficient label for reports; do not use the label as field identity.

### `make_problem`

```python
make_problem(order, defaults, compiled)
```

Build a callable subsystem using selected equations/variables and a frozen background state. Module settings determine which dimensions are analyzed.

### `make_problem.F`

```python
make_problem.F(x)
```

Insert trial coordinates into a copy of the frozen background, evaluate selected RHS components and return NaNs for invalid trial evaluations.

### `numerical_jacobian`

```python
numerical_jacobian(F, x)
```

Approximate derivatives of F by centred finite differences with a scale-aware coordinate step; this is a local numerical Jacobian, not symbolic differentiation.

### `newton`

```python
newton(F, x0)
```

Damped Newton solve from x0 using finite-difference Jacobians and least squares for singular systems. Return (point, residual, iterations), or None on failure.

### `grid_axes`

```python
grid_axes(var)
```

Build starting-point coordinate arrays from DEFAULT_GRID and per-variable overrides. These are Newton guesses, not the PDE's physical mesh.

### `is_same_root`

```python
is_same_root(a, b)
```

Compare a candidate against accepted roots using the configured merge tolerance to avoid duplicate multistart results.

### `main`

```python
main(path: Path, output_dir=None)
```

Load the chosen equation/provider file, run the configured multistart search and write root CSV/NPZ files into output_dir.

## engine/numerics/ode.py

[Открыть исходник](../engine/numerics/ode.py)

### `_check_expression`

```python
_check_expression(tree: ast.AST, allowed_names: set[str], size: int)
```

Inspect the expression AST against the allowed syntax before compilation; this is the restricted legacy equation interface, not arbitrary Python execution.

### `load_equations`

```python
load_equations(source: str)
```

Read ORDER/default parameters and RHS expressions, or detect a provider descriptor. Validate and prepare the representation used by the numerical caller.

### `_compile_profile`

```python
_compile_profile(expression: str, parameters: dict, dimension: int)
```

Compile an allowed initial-profile text expression; coordinates and parameters are supplied during projection. Callable profiles are handled by the caller.

### `project_initial_fields`

```python
project_initial_fields(profiles: dict, order: list, parameters: dict, quadrature_points: int)
```

Numerically project configured physical initial fields into the ORDER cosine modes. This initializes coefficients, not a simulation step.

### `integrate`

```python
integrate(source: str, initial: dict, overrides: dict, t_start: float, t_end: float, method: str='Radau', rtol: float=1e-07, atol: float=1e-09, max_step: float=np.inf, save_dt: float=0.1, first_step: float | None=None, initial_fields: dict | None=None, quadrature_points: int=64)
```

Integrate legacy generated equations with adaptive solve_ivp and assemble a standard archive dictionary; caller decides when to write it.

### `integrate.derivative`

```python
integrate.derivative(t, state)
```

Evaluate compiled legacy equations for the current state/time and reject nonfinite derivatives before the solver proceeds.

### `integrate_provider`

```python
integrate_provider(provider, initial, t_start, t_end, method='Radau', rtol=1e-07, atol=1e-09, max_step=np.inf, save_dt=0.1, output_times=None, integral_clock=None, integration_session=None)
```

Additional universal callable-RHS path; legacy integrate() stays unchanged.

Randomness and structural changes must be handled by the caller BETWEEN
integrations. A provider's order and RHS are immutable throughout this call.
Returned keys match the scalar-equation path. By default use solve_ivp;
an optional segment-owned session retains adaptive history across slabs.
Never share a session between different providers or event maps.

### `integrate_provider.rhs`

```python
integrate_provider.rhs(t, y)
```

Evaluate the provider's coefficient RHS and, when enabled, append the auxiliary integrated-hazard derivative. The clock coordinate is not a physical field.

### `integrate_provider.stop`

```python
integrate_provider.stop(t, y)
```

Return accumulated hazard minus its threshold for upward terminal root detection; do not draw randomness during root finding.

## engine/numerics/rhs.py

[Открыть исходник](../engine/numerics/rhs.py)

### `is_provider_source`

```python
is_provider_source(source)
```

Identify JSON provider-descriptor text by its leading non-whitespace character; legacy equation text follows the other path.

### `load_provider`

```python
load_provider(source)
```

Parse a JSON descriptor and restore the trusted Galerkin provider, including model-source compatibility checks.

### `evaluate_rhs`

```python
evaluate_rhs(compiled, t, state, parameters)
```

Shared adapter for existing Newton/spectrum; evaluate vector RHS only once.

## engine/numerics/spectrum.py

[Открыть исходник](../engine/numerics/spectrum.py)

### `_check_expression`

```python
_check_expression(tree, allowed_names, size)
```

Inspect the expression AST against the allowed syntax before compilation; this is the restricted legacy equation interface, not arbitrary Python execution.

### `load_equations`

```python
load_equations(source)
```

Read ORDER/default parameters and RHS expressions, or detect a provider descriptor. Validate and prepare the representation used by the numerical caller.

### `label`

```python
label(order, i)
```

Format a stored field/mode entry into a readable coefficient label for reports; do not use the label as field identity.

### `make_problem`

```python
make_problem(order, defaults, compiled)
```

Build a callable subsystem using selected equations/variables and a frozen background state. Module settings determine which dimensions are analyzed.

### `make_problem.F`

```python
make_problem.F(x)
```

Insert trial coordinates into a copy of the frozen background, evaluate selected RHS components and return NaNs for invalid trial evaluations.

### `numerical_jacobian`

```python
numerical_jacobian(F, x)
```

Approximate derivatives of F by centred finite differences with a scale-aware coordinate step; this is a local numerical Jacobian, not symbolic differentiation.

### `newton`

```python
newton(F, x0)
```

Refine a proposed point with damped Newton; return point/residual/iteration count or raise RuntimeError on nonconvergence/nonfinite data.

### `fmt_complex`

```python
fmt_complex(z)
```

Format complex eigenvalues consistently for human-readable reports, suppressing negligible imaginary parts.

### `main`

```python
main(path: Path, output_dir=None)
```

Load the selected subsystem, optionally refine its point and export the Jacobian eigenvalues/eigenvectors and analysis metadata.

## engine/simulation/grid.py

[Открыть исходник](../engine/simulation/grid.py)

### `Grid.__init__`

```python
Grid.__init__(self, shape, lengths)
```

Validate 1D/2D/3D shape and positive lengths, then construct cell centres, spacings and one-cell volume used in integrals.

### `Grid.boundaries`

```python
Grid.boundaries(self, spec)
```

Expand one BC or per-axis low/high pairs; periodic must be paired.

### `Grid.gradient_faces`

```python
Grid.gradient_faces(self, u, bc, axis)
```

Interior differences and exact half-cell Dirichlet distances.

### `Grid.face_average`

```python
Grid.face_average(self, a, axis, periodic)
```

Compute the two-cell arithmetic mean used to place cell values on shared faces; preserves NumPy broadcasting.

### `Grid.permeability_faces`

```python
Grid.permeability_faces(self, permeability, axis, is_periodic)
```

Harmonic face coefficient: a zero cell blocks both adjoining faces.

### `Grid.transport`

```python
Grid.transport(self, mu, mobility, bcs, permeability=None)
```

Compute div(L grad(mu)); L is (N,N,*shape), may depend on U.

Mixed topologies with off-diagonal transport are deliberately rejected:
a cross-flux cannot be periodic for one field and open for another.

## engine/simulation/solver.py

[Открыть исходник](../engine/simulation/solver.py)

### `Simulation.__init__`

```python
Simulation.__init__(self, config, model=None, handlers=None, stochastic=None)
```

Validate grid/time/seed, create the run-owned RNG, registry and initial fields, and prepare scheduled events. Initial data is copied into physical arrays.

### `Simulation._load_model`

```python
Simulation._load_model(config)
```

Delegate configured model creation to the trusted model loader; kept separate so initialization can also accept a supplied Model instance.

### `Simulation.check_parameters`

```python
Simulation.check_parameters(p)
```

Require finite scalar field parameters, nonnegative diffusion and positive capacity. This checks metadata, not numerical stability of a chosen PDE.

### `Simulation.add_field`

```python
Simulation.add_field(self, name, family, values, t, parameters, parent=None, bc=None, label=None)
```

Append a unique-ID registry row and physical field after validating parameters/BC/data. Never renumber retired rows or existing field IDs.

### `Simulation.index`

```python
Simulation.index(self, name)
```

Resolve a stable field ID to its global registry row, including retired history; unknown IDs raise ValueError.

### `Simulation.next_name`

```python
Simulation.next_name(self, prefix)
```

Find an unused legacy prefix_integer name while reserving all historical IDs, including inactive fields.

### `Simulation.target_indices`

```python
Simulation.target_indices(self, targets)
```

Resolve explicit IDs or the legacy wildcard ['*'], returning only currently active global registry indices.

### `Simulation.retire`

```python
Simulation.retire(self, i, t)
```

Mark a field inactive with a death time and zero its current physical values, retaining historical identity in the registry.

### `Simulation.record`

```python
Simulation.record(self, event)
```

Assign a deterministic event ID, attach it to newly created fields and append a serialized event record.

### `Simulation.apply`

```python
Simulation.apply(self, event)
```

Resolve semantic targets, invoke the existing event handler, then record its resulting metadata. Handlers may mutate state and registry.

### `Simulation.deterministic_step`

```python
Simulation.deterministic_step(self, t, h)
```

Advance active grid fields by explicit Heun over h. Reject nonfinite/materially negative values; only roundoff-scale negatives are clipped.

### `Simulation.scheduled`

```python
Simulation.scheduled(self, t)
```

Apply queued scheduled events up to time t in stored order and advance the schedule cursor once per event.

### `Simulation.snapshot`

```python
Simulation.snapshot(self, t)
```

Save a copy of the current physical fields and cheap diagnostics at t; later births do not resize earlier snapshots immediately.

### `Simulation.run`

```python
Simulation.run(self)
```

Execute the single-use FV loop, respecting event/save boundaries and held interventions; return its standard archive dictionary.

### `Simulation.archive`

```python
Simulation.archive(self)
```

Pad history ONCE at serialization; no repeated expansion of snapshots.

## engine/visualization/archive.py

[Открыть исходник](../engine/visualization/archive.py)

### `resolve_path`

```python
resolve_path(value)
```

Expand a user path and resolve it relative to the current working directory, not the package location.

### `read_archive`

```python
read_archive(path)
```

Read the legacy cosine archive layout and return time/coefficient/field/mode/length arrays. Other representations use plot_archive's dispatch.

### `make_reconstructor`

```python
make_reconstructor(fields, modes, lengths, view_dimension, grid_points, slice_fractions=SLICE_FRACTIONS)
```

Prepare legacy cosine reconstruction for the requested view. Return coordinates, field-to-column masks and a callback taking (coefficient row, field ID).

### `make_reconstructor.reconstruct`

```python
make_reconstructor.reconstruct(frame, field)
```

Reconstruct the selected field's stored coefficients on the prepared plotting coordinates; never advance the simulation.

### `field_limits`

```python
field_limits(coefficients, modes, field_indices)
```

Bound each field by its zero-mode baseline plus/minus the absolute amplitudes of nonzero cosine modes over saved frames, then pad the range for plotting.

### `plot_archive`

```python
plot_archive(path, view_dimension=None, grid_points=80, frame_index=None, save=None, fps=12, color_scale='frame', slice_fractions=SLICE_FRACTIONS)
```

Inspect representation and route to spectral, grid or legacy cosine rendering. Cell averages must never be interpreted as Fourier coefficients.

### `plot_archive.draw`

```python
plot_archive.draw(index)
```

Render one saved legacy frame, updating titles/color limits and artists for the animation or image exporter.

## engine/visualization/spatial.py

[Открыть исходник](../engine/visualization/spatial.py)

### `validate_view`

```python
validate_view(shape, dimension, grid_points, slices)
```

Validate requested display dimension, sampling resolution and slice fractions against the archive shape; return the effective view dimension.

### `slice_indices`

```python
slice_indices(shape, fractions)
```

Convert fractional slice locations into bounded cell indices for each spatial axis; these are display choices only.

### `draw_field`

```python
draw_field(ax, u, coords, lengths, dimension, limits, grid_points=80, slices=(0.5, 0.5, 0.5), cell_average=False)
```

3D: three coloured orthogonal planes at cell-centre coordinates.

2D shows XY at the chosen Z; 1D shows X at the chosen Y/Z.
grid_points caps only the rendered 3D planes, never solver resolution.

## engine/visualization/spectral.py

[Открыть исходник](../engine/visualization/spectral.py)

### `ArchiveSequence.__init__`

```python
ArchiveSequence.__init__(self, path)
```

Index saved times in manifest order, retaining duplicate event-boundary frames and an optional terminal state. Keep only one reconstructed archive cached.

### `ArchiveSequence.frame`

```python
ArchiveSequence.frame(self, index)
```

Return field-ID to grid-array values, Grid, actual time and segment ID for one frame. Read terminal physical values directly; otherwise reconstruct stored coefficients.

### `ArchiveSequence.select_time`

```python
ArchiveSequence.select_time(self, requested_time, mode='nearest', event_side='after')
```

Return a saved-frame index for a finite model time, without interpolation.

nearest chooses the closest time (earlier time wins equal distances).
before/after choose the last/first saved time on that side of the request.
At duplicated event times event_side chooses the first/last frame in
manifest order. 'after' is the default, including terminal-state frames.
Requests outside the saved time range fail instead of silently clamping.

### `plot_spectral`

```python
plot_spectral(path, view_dimension=None, grid_points=80, frame_index=None, save=None, fps=12, color_scale='frame', slice_fractions=(0.5, 0.5, 0.5), time=None, time_mode='nearest', event_side='after')
```

Render stored fields; select by index OR model time, otherwise animate.

New time options are trailing arguments to preserve existing callers.
Returns selection metadata for a still frame, None for animation. No PDE
solver or model source is loaded. The title always reports the actual time.

### `plot_spectral.draw`

```python
plot_spectral.draw(k)
```

Draw one stored frame using actual time and field presence; keep one colorbar per field and never interpolate between segments.

## engine/visualization/viewer.py

[Открыть исходник](../engine/visualization/viewer.py)

### `plot_grid_archive`

```python
plot_grid_archive(path, view_dimension=None, grid_points=80, frame_index=None, save=None, fps=12, color_scale='frame', slice_fractions=(0.5, 0.5, 0.5))
```

Render the cell-average archive representation without a spectral transform; select a saved index or animate stored snapshots.

### `plot_grid_archive.draw`

```python
plot_grid_archive.draw(k)
```

Extract and display a saved grid frame, retaining stable field identity and updating labels/colorbars.
