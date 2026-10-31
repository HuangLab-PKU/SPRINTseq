# Standard artifact checks and recovery in SPRINTseq

## Placement in the pipeline

Image/acquisition checks belong **before readout**. Reaction history usually needs molecular evidence, but detection planning need not wait for a definitive mechanism call. Cover the codebook under plausible image-supported erasures and pause histories before committing to a detection window. Use a small pilot when mechanism resolution is necessary; do not infer chemistry from a dark region alone.

The standard order is:

1. Source inventory, axis roles, registration record and bounded image-validity prescan.
2. Codebook-wide detection coverage under normal, erased-observation and candidate-history scenarios.
3. Original requested detection rounds over their original footprint; supplemental rounds only in image-derived regions requiring them, with an uncertainty margin. If normal-region code coverage itself is missing, the needed supplemental round is global.
4. Full measured intensity readout for detected coordinates. Missing/nonfinite reads stop execution; they never become zero.
5. Baseline decoding and **all-panel** post-decode audit. Standard transcript export may preserve provisional results for repair, but an audit HOLD prevents density publication and prevents workflow completion.
6. Automatic source-bound mechanism qualification, observed-dimensional inference, local measured spot recovery, independent candidate validation and transactional default activation.

This separates prevention from recovery. A first pause can move a code's first bright base from acquisition four to acquisition five. A rounds-1–4 detector then has no measurement capable of discovering that molecule. Correcting labels of existing points cannot recover it. The new preflight adds the required coverage in advance. An already processed run uses local supplemental discovery and intensity readout, preserving old IDs; a full-slide readout rerun is not the default recovery.

## Failure taxonomy

### Missing acquisition or rendered observation

Evidence: expected acquisition plan, source-hashed completion evidence, cycle/channel/tile/Z inventory, renderer support. An absent entire stack after confirmed completion differs from an unfinished acquisition, an empty frame or a gapped Z-stack. A missing whole mosaic plane stops the standard readout. Local lack of support remains NoData; a real overlapping donor remains valid. Never substitute dark intensity, a neighboring expression count or an invented registration transform.

Use `spatial_img_core.observation_qc.census` before image processing, native registration observability gates, and exact renderer support at the point stage. Legacy runs lacking these records report unavailable coverage rather than claiming a clean inventory.

### Optical loss or background failure

Evidence: same-channel cross-cycle attenuation, contrast and high-frequency texture after robust exposure normalization. Image masks are independent of decoded gene labels. Keep uncertainty at boundaries and record pyramid resolution. Recover focus from source Z data when possible; otherwise exclude unreliable dimensions. A black patch is not itself proof of a paused reaction.

Image channel/cycle roles must come from metadata, **not array position**. A stored `cy5,cy3` axis is valid and must be reordered into the requested `cy3,cy5` analysis order. Duplicate or missing roles hold execution. `prescan` and `points` implement this contract; historical positional-QC sources are retained for provenance and must not be reused as newly qualified validity.

### Geometry or cross-round correspondence failure

Evidence: native per-tile registration diagnostics, a solved registration report bound to the current shift-table hash, and independent source/mosaic acceptance. Unresolved transforms or mismatched report signatures hold. A numeric solved graph is not proof of pixel accuracy; quality exceptions require separate qualification. Do not enlarge the intensity search radius until nearby molecules can be mistaken for the same point.

### Reaction-history change

Evidence: complementary observed-cycle holdouts, unique-code anchors, normal controls, split transcript-ID groups and spatial coherence. Distinguish pure optical erasure, paused reaction and mixed/insufficient evidence. The detection planner can cover more hypotheses than the currently qualified decoder can resolve; such regions remain explicitly held at molecular acceptance.

A qualified region can support a history, but individual contradictory evidence can veto it. Retain within-history collisions and missing terminal observations as unknown. Neither increased Q20 yield nor smoother density selects a mechanism.

### Detection-window blindness

Evidence: project the exact codebook through each candidate history and count observable bright dimensions in the requested discovery rounds. Audit every feature, including features with zero detected points. Add needed rounds in the affected spatial footprint before readout. `coverage_plan` emits the effective rounds, affected pattern masks, at-risk features and irrecoverable feature-pattern combinations.

If already processed, use the [measured detection recovery stage](../observation_repair.md#reaction-history-can-invalidate-the-original-detection-window). Discovery stays label independent. Requiring two *late discovery rounds* creates a new code-specific bias and is not the original detection contract.

### Duplicate, displaced or shoulder detections

Evidence: actual source peaks in expected-bright discovery channels, native localization tolerance, separation from existing points and deduplication among candidates. High decoder probability does not prove a unique physical molecule. Reject a shoulder of a brighter neighboring peak; do not let unrelated dark-code-channel noise qualify it. Read real intensities at accepted coordinates and verify source replay on old points.

### Ambiguous or model-inadequate observations

Evidence: held-out intensity contradictions, background competition, projected-code collisions, insufficient independent support, or failed technical controls. Test response calibration or alternative histories using independent controls; do not optimize a desired gene count. If no tested correction is supported, retain uncertainty and a spatial/feature limitation record. Low observed counts in such a region do not establish biological absence.

### Unsynchronized outputs

Evidence: source bindings, stable transcript IDs and agreement between modern/legacy readers. Update transcripts, intensities, assignments, matrices, density and maps as one version. Appending points requires new IDs and matching sidecars; old CSV rows should remain byte-identical when they are not meant to change. Preserve baseline snapshots and validate a transactional activation.

## Executable standard entrypoints

`experiments/run_pipeline.ps1` now invokes `sprintseq.observation_checks.workflow`. It checks this invocation's native exit receipt, because an environment launcher can mask a Python exit failure. The Python supervisor owns child waits, logs, exit receipts, source-validated checkpoints and a writer lock.

```powershell
.\experiments\run_pipeline.ps1 -RunId <run-id> -RefFile <exact-codebook.csv>

# Equivalent native supervisor
python -m sprintseq.observation_checks.workflow --run-id <run-id> --ref-file <exact-codebook.csv>

# Inspect sources without executing readout
python -m sprintseq.observation_checks.preflight --run-dir <processed-run> --ref-file <exact-codebook.csv>

# Explicit forensic audit of an existing run; report-only does not grant acceptance
python -m sprintseq.observation_checks.audit --run-dir <processed-run> --ref-file <exact-codebook.csv> --report-only
```

The default `sprintseq readout` entry calls the preflight itself, accepts `--ref-file`, and otherwise requires the run's codebook snapshot. Both the readout CLI and library entry refuse to overwrite an existing readout. The supervisor reuses it only through a matching checkpoint. Accepted default revisions require versioned recovery, not ordinary readout/gene-calling overwrite.

The default `sprintseq gene-calling` CLI verifies its readout binding, rejects stale mapping output after decoder failure and writes the panel audit. Exit 2 denotes a valid provisional decode with a scientific HOLD. The standard supervisor exports the provisional table, automatically runs qualified observation repair and supplemental discovery, and validates candidate density/counts before default activation. The audit flag is not treated as a successful clean decode. Direct legacy scripts calling the old gene-calling implementation are not substitutes for this standard supervisor.

The initial integrated protocol qualification is ten sequencing rounds and two SBS channels (`cy3`, `cy5`). Other protocols need an explicit qualified adapter. The prescan requires finalized OME-TIFF with usable axis metadata and a bounded pyramid; unsupported layouts hold rather than silently skipping QC. This does not replace the upstream raw-acquisition census or native image-processing acceptance.

## Reports and dispositions

`observation_checks/current_preflight.json` points to a content-bound preflight containing image/codebook/source-code identities, ordered axis mapping, image-validity masks and supplemental detection ROIs. `readout_binding.json` binds the actual readout files to that plan. `post_decode/` contains every panel feature's count, all feature-region comparisons and a receipt with recovery actions. `workflow_state.json` and per-attempt logs distinguish RUNNING, HOLD, FAILED and COMPLETE_WITH_REPORTED_COVERAGE_LIMITS.

Post-decode screening uses image-defined components, nearby image-valid controls, effect-size checks and multiplicity-adjusted count-comparison p-values. These are **triage statistics**, not biological false-discovery estimates: local biology can cause enrichment or depletion. The tool emits all features, not just flagged ones, and never changes a gene based on a density ratio. Known residual-feature records also force HOLD.

Local unobservable feature-pattern combinations remain NoData even if no spots were detected there. Planning continues for observable regions, while molecular acceptance remains held. Histories outside decoder qualification, insufficient controls, known residuals and registration exceptions cannot disappear behind a successful command exit.

Recovery is scoped to the failed boundary: restore/qualify acquisition and geometry; regenerate correctly bound validity; qualify the mechanism; supplement local discovery; or retain unknowns. Repeat only source-matching unfinished stages. Do not lower thresholds, erase inconvenient genes or replace unknown measurements with biological zeros to clear a gate.

## Acceptance

Require source-faithful measurements, exact preservation outside the authorized change, all-panel review, independent count reconstruction, consistent modern/legacy readers and synchronized outputs. Technical control disagreement rates are conditional on their fitted-model/reference assumptions; they are not universal biological accuracy. A reported unresolved feature remains unresolved even when other genes improve.

## Reviewed exceptions and reliable prefixes

A review must distinguish sampling/control mismatch, physically supported repair
and remaining non-identifiability. Morphology/cell-mask area is independent of
the decoded gene count; zero assigned spots alone does not establish absent
tissue. Record all item-level evidence and preserve unresolved features.

The qualified history-event limit concerns events that can affect an observed
cycle. Strictly later unmeasured events are marginalized out rather than making
an already measured prefix unusable. The corresponding recovery is tested on
both positive technical references and background controls.

After a validated repair, reapply the original cell-assignment threshold to the
repaired IDs using the same geometry. Coordinate/intensity edits require their
own source-peak and localization-control evidence. Biological FDR and complete
artifact elimination are not implied by these technical acceptance checks.

## All-spot discovery and mixed-observation recovery

An all-spot cycle supplies measured coordinates; it is not an additional barcode
base. Bind its role to the acquisition protocol, and qualify alignment against
existing trusted points in each affected region. Merely having an extra image
plane is insufficient. Check anchor support, displacement tails, actual image
peaks and separation from every existing point before admitting additions.

A missing early sequencing acquisition can affect both discovery and decoding.
An acquisition missing after the original discovery window primarily affects
decoding. Import already qualified partial-observation points before attributing
an entire blank rectangle to failed detection. New late/all-spot candidates must
have code-derived discovery-loss support, not simply a desirable gene label or
a higher total count. Retain old IDs and append new globally unique IDs with
source-tier and original-index provenance. Keep unmeasured intensity dimensions
as NaN and retain exact observed-dimension likelihoods from the source-fitted
model; do not retrain a full-dimensional model on missing-value placeholders.

If the all-spot image itself lacks useful signal or alignment, it cannot rescue
that region. Try valid measured later sequencing rounds under the same source,
codebook and history checks. A more restrictive candidate screen, such as
requiring expected bright peaks in two distinct observed rounds, can reduce
recall for particular codewords and must be reported as a conservative pilot,
not silently adopted as the universal discovery contract. Failed support is a
recorded recovery limitation, not permission to interpolate density.

Three checks prevent misleading apparent repairs:

- A noise-only Z-stack can have a high normalized high-frequency ratio. Require
  structured signal before selecting a focus plane; all-zero focus scores mean
  no qualified focus plane, not that the first Z plane is usable.
- Blank canvas outside every acquired sequencing plane is not a missing-tile
  region. Exclude it from acquisition-footprint planning, while keeping any
  detected points there explicitly unsupported. A source-specific declared
  missing-tile mask must retain its original qualified contract even if a
  renderer offers an overlapping donor; changing that contract needs separate
  qualification.
- Observed-only QC statistics exclude declared NaN values and report measured
  and missing counts separately. Empty measurement sets remain unknown, and
  infinity is an error. `observation_repair.mixed_qc` implements this boundary.

Before default activation, reconcile the mixed matrix against independently
verified partial-tier matrices plus the scoped complete-domain repair and new
spot contributions. Preserve pixel-based cell coverage when available, alongside
spot-derived coverage. Verify all density planes against direct counts. Both old
and newly imported native CSV row text must survive staging without floating
point reserialization; otherwise apparently equivalent coordinates can fail
exact Parquet/legacy parity. A detached candidate pointer allows full preparation
without changing the current reader entrypoint. Preserve archived source paths
and require dependent exports to rebind to the accepted generation.


## Standard automatic recovery entry

Use the installed standard entry for new runs:

```powershell
sprintseq pipeline --run-id <run-id> --ref-file <exact-codebook.csv>
# Equivalent launcher with invocation-specific native exit checking:
.\experiments\run_pipeline.ps1 -RunId <run-id> -RefFile <exact-codebook.csv>
```

`--recover-existing` resumes a historical, not-yet-activated readout without
re-running full-slide discovery. `--candidate-only` builds and verifies a
versioned candidate without default cutover. `--audit-only` explicitly requests
the former stop-at-audit behavior. Automatic recovery is the normal default.
`--allspot-cycle` defaults to11 for the qualified ten-round profile; availability
and regional alignment are checked, so missing/unusable C11 does not block the
later-round route. C11 is used only to measure coordinates, never barcode bases.

The executable stages are:

1. Before readout, verify acquisition completion when a source plan is provided,
   registration bindings, finalized image axes, exact rendered support and
   image-only attenuation/texture masks. `coverage_plan` adds code-derived later
   discovery rounds in affected regions before the initial readout.
2. Export the provisional decode and audit every codebook feature. Statistical
   density flags select review, never a requested gene count or a smooth image.
3. Run exact point coverage, fit using only fully observed training candidates,
   verify full-observation numerical parity and positive/negative history
   controls, qualify optical versus reaction-history policies, and marginalize
   unreliable dimensions. Missing files never imply a chemical pause.
4. Discover additional measured points in early-failure components. Try the
   all-spot cycle, qualify regional/local alignment, then automatically try
   available rounds after the original discovery window when all-spot support is
   missing or fails. Preserve the original detector recipe, deduplication and
   exclusion from old points. Verify exact old-intensity replay, native-pixel
   renderer support, actual discovery/expected-sequence peaks, code-derived
   discovery loss, posterior and history confidence. Unqualified blocks remain
   explicitly unresolved. Insufficient anchors cannot be repaired by density
   filling. The conservative late-round two-observed-cycle peak gate may lose
   sparse-codeword recall and is recorded as such.
5. Build a complete candidate: transcripts, observed intensities/validity,
   native CSV views, all-feature density and QC. If supported geometry exists,
   refresh scoped assignments at its recorded threshold, independently recount
   matrices and rebuild the standard existing maps. Do not select a segmentation
   model merely to repair readout. Incomplete/nonstandard geometry prevents a
   default cutover and leaves the molecular candidate for diagnosis.
6. Re-audit the complete panel, retain residual flags/unknowns, verify exact
   outside-scope preservation and native/Parquet parity, then publish through a
   rollback-capable transaction. Verify default readers after switching and keep
   baseline directories plus relocation receipts. Candidate tampering, changing
   inputs or failed readback cannot be accepted. Custom derivatives not rebuilt
   are explicitly listed as archived-only products rather than left stale.

The workflow is `sprintseq.observation_checks.workflow`; automatic orchestration
is `observation_checks.recovery`; reusable discovery and policy gates live in
`observation_repair.discovery` and `discovery_policy`. Candidate construction and
transactional output synchronization belong to `spatial_cells.observation_candidate`
and `observation_publish`. None requires a slice list, a selected gene, or an
import from a dated experiment directory.

A raw plan, when available, lives at
`observation_checks/acquisition_plan.json`, with `run_id`, `raw_run`,
`acquisition_complete: true`, hash-bound `completion_evidence`, and
`expected_observations` entries containing cycle/channel/tile/render-tile/Z
indices. The census distinguishes whole-stack absence, unfinished acquisition,
empty files and gapped stacks. Without a raw plan the receipt explicitly says
raw completeness is unavailable; it still checks image/rendered support and
does not invent a clean raw inventory. Missing whole mosaics, unresolved
registration, invalid source identities and failed model controls remain
explicit actionable exceptions. This method repairs decoding/discovery from
remaining measurements; it does not manufacture an in-focus raw stack.

Records are `observation_checks/workflow_state.json` and
`observation_repair/automatic_<source-key>/`: child command/exit logs, molecular
controls, per-component all-spot/late attempts, all-panel post-recovery audit,
`resolution.json`, candidate and activation receipts. Completed accepted defaults
are source-verified without duplicate fitting or additions. Technical completion
retains residual uncertainty and does not assert biological FDR or universal
artifact elimination.


### Re-entry after downstream exports

Read-only verification of an already accepted generation permits additional
artifacts. All original recorded products must still match. The one supported
retirement exception is `density_Q20_all.tif`: it can be absent when a
`density_groups*.json` receipt binds every original per-gene density, every
replacement output hash, and TIFF labels that uniquely partition the current
panel. Unexplained missing stacks, changed counts/intensities/assignments and
changed per-gene density remain failures. `spatial_cells.observation_reentry`
implements this check; cutover still uses the strict transaction inventories.
Historical archive revalidation is separate from this current-output check.

Where host policy disallows unsigned network PowerShell scripts, use the
installed `sprintseq pipeline` entry (or `python -m sprintseq.cli.main pipeline`).
It invokes the same standard workflow without changing the host signing policy.
