# Observation failure detection and repair

Mandatory pre-readout prevention and all-panel acceptance are specified in [Standard artifact checks and recovery](methods/artifact_control.md). The repair algorithms below are qualified recovery stages within that flow.

The opt-in post-readout stage detects spatially local image failures, carries explicit per-cycle/channel validity into decoding, compares acquisition histories, and writes a versioned molecular dataset. It does not manufacture pixels, missing bases, spots or cell masks.

See [Method specification](methods/observation_repair.md) for the statistical model, mechanism decision, evaluation protocol and package ownership. The historical experiment package is not a runtime dependency.

## Pipeline entry

```powershell
mamba run -n spatial sprintseq observation-repair `
  --run-dir <processed-run> `
  --output-root <processed-run>/observation_repair/<new-generation> `
  --publish
```

An existing exported model may be supplied with `--model-dir`. Without one, the stage exports a new 60-iteration deterministic-recipe PoSTcode fit from the actual intensity table and checks full-observation numerical parity; this is not recovery of unsaved historical fitted parameters. CPU fitting is default; `--fit-device auto` opts into installed backend selection.

Requirements: finalized source OME-TIFF; exact codebook; standard transcripts and intensity tables with identical IDs; accepted same-frame cell mask, positions and assignment. Install `spatial-img-core[observation-qc]`, `sprintseq` and `spatial-cells` in the same mamba environment. Entry points live in their owning packages:

- `python -m spatial_img_core.observation_qc.points`: source-bound image validity.
- `python -m spatial_img_core.observation_qc.census`: completed-acquisition missing-stack inventory and native declaration overlay.
- `python -m sprintseq.observation_repair`: masked history-aware molecular inference.
- `python -m sprintseq.observation_repair.qualify`: unaffected-control erasure and injected-history qualification.
- `python -m spatial_cells.observation_revision`: standard transcript/assignment writers, matrix and native density/stack rebuilding.

The unified entry owns actual child waits, logs and exit receipts. One per-run exclusive lock prevents duplicate writers. A valid existing checkpoint is reused only after current source/hash checks; failed partial directories are preserved and are not blindly overwritten. All ordinary processing happens in scripts, without model-driven polling.

## Observation contract

The image module computes per-cycle/channel attenuation and texture-loss masks using cross-cycle same-channel references and robust exposure normalization. Masks depend on imagery, not decoded labels or Q20 yield. Boundary pixels remain explicitly uncertain. The recorded pyramid resolution limits edge precision; sample independent normal/affected regions when qualifying a new acquisition type.

Reason bits are ATTENUATION=1, TEXTURE_LOSS=2, SOURCE_MISSING=4 and BOUNDARY_UNCERTAIN=8. `point_validity.npz` binds every transcript ID to an observed[T,C] mask and reason codes. A channel is excluded only where its own measurement is unreliable. Both-off is a biological code symbol only when both channels are actually observed.

Modern finite-render support is sampled exactly at level0 pixel centres using `floor(x+0.5)` and the renderer's little-endian row-packed masks. A missing tile does not invalidate a true overlapping neighbor measurement. Missing-source reasons never generate a chemical-pause hypothesis. Legacy images without these support records report coverage unavailable; a black pixel alone does not prove a missing tile.

Before focal stacking, `observation_qc.census` consumes an acquisition-resolver plan with run_id, acquisition_complete, source-hashed completion_evidence and expected_observations (cycle, channel, tile_id, render_tile, z_indices). A completed but absent stack generates explicit `register.declared_missing_observations`. An unfinished scan, empty file or partial/gapped Z stack is a HOLD. Native registration still rejects wholly unobservable acquisition groups rather than inventing transforms. After rendering, exact support propagates to molecular validity. The census adapter emits an overlay; it does not hot-edit another running pipeline's configuration.

## Decode and uncertainty

Unreliable dimensions are marginalized from the fitted multivariate Gaussian; covariance and all target/background/infeasible classes are retained. For both-channel image failures the decoder considers unchanged history and up to two paused acquisitions. A pause consumes no base; later acquisitions move to the next unconsumed base, and terminal bases remain unobserved. A four-state nuisance mixture represents each unobserved paused acquisition. Background means stay unchanged.

Probabilities are evaluated with pause priors0.1,0.5,0.9. A gene must agree across them; confidence, second probability and entropy come from the least-confident case. Exact cross-history projected-code collisions, no-evidence points and patterns beyond the qualified event limit become UnknownObservation (non-target, Q0), with reasons retained. These scores are not biological FDR estimates.

Qualification uses8,192 unaffected high-confidence technical-reference points (or all available when fewer). Conditions include first-cycle pause, single-channel source absence, both-channel optical erasure, unknown history, and injected single/double pause. Every condition must retain at least50 Q20 reference calls with at most1% disagreements before publication. Reference labels and injected histories do not establish biological truth; low retention is explicitly reported rather than repaired by relaxing thresholds.

## Versioned output and consumption

`<generation>/revision/` contains:

- `readout/transcripts.parquet`, written through the standard schema API;
- `segmented/transcript_assignment_cellsam.parquet`, referencing the unchanged original mask;
- `segmented/cell_gene_matrix_Q20_panel202.csv`, preserving every original cell and panel feature;
- `readout/density_Q20/` and `readout/density_Q20_all.tif`;
- `segmented/cell_observation_qc.parquet`, preserving missing/no-measurement status beside counts;
- `receipt.json`, including source identities, hashes and executable validation outcomes.

The publisher verifies every assigned mask pixel, independently reconstructs matrix counts, reads and compares all density planes against revised points, validates the stack and ensures unaffected transcript values remain unchanged. `publication_manifest.json` hashes every product. With `--publish`, `<processed-run>/observation_repair/current.json` is atomically updated to this revision. Baseline files remain in place; previous pointer generations are retained. Consumers must explicitly resolve this pointer or use the revision path. Legacy commands directed at the baseline `readout/` continue reading that baseline, and the other session's frozen runs are not silently redirected.

Unmeasured observations are not biological zeros. Count matrices remain observed counts with a coverage sidecar; `observed_fraction` is derived from detected spots, not complete cell pixels. The tool cannot restore spots missed in every detection cycle, nor certify every local geometric distortion from these image metrics. Such limitations and unknown calls remain in the release receipt.

## Region-qualified mechanism policy (v3)

The unified pipeline now inserts `sprintseq.observation_repair.mechanisms` between image validity and repair. `--qc-dir` can reuse a previously source-validated immutable image-QC generation without duplicating its masks.

Image-derived connected components propose failure regions. No-shift support is determined from prefix-only identities with unique projected codewords and P>0.99, later observed cycles held out from identity selection, unaffected normal controls, disjoint transcript-ID halves and2048px spatial blocks. The fixed gates require at least100informative affected points,30controls and99%prefix/reference agreement in controls; pause preference must be <=15% in affected/control sets and both affected halves. At least3local blocks must also pass their support/coherence gates. Only those qualified local blocks disable pause alternatives. A pause-favoring, first-cycle, mixed-history or insufficient-data region retains the original uncertainty-aware mixture. More Q20calls or a smoother density map is never a mechanism-selection criterion.

The policy is an immutable transcript-ID/cycle boolean array with source/model/QC digests and all regional evidence. Decode refuses scope, ID, source or eligibility mismatches. It changes allowable histories, never intensity, coordinates, validity masks or confidence thresholds. The v3 receipts include this policy. Missing-source observations remain ineligible for chemical-pause inference.

Per-run qualification results and source-specific figures belong in the canonical experiment README. Validate locality, independent optical-only agreement and unchanged density outside the supported policy. The qualified implementation currently targets ten sequencing cycles and two SBS channels; other layouts need explicit qualification.


## Explicit default-directory activation

A versioned `--publish` is distinct from replacing default directories. After explicit authorization, run:

```powershell
mamba run -n spatial python -m spatial_cells.observation_activation prepare --run-dir <processed-run>
mamba run -n spatial python -m spatial_cells.observation_activation activate --run-dir <processed-run>
mamba run -n spatial python -m spatial_cells.observation_activation verify --run-dir <processed-run>
```

Preparation builds modern and legacy tables, preserves the original all-call matrix semantics, adds explicit Q20 aliases, regenerates QC and any pre-existing cell/spot maps, and checks modern/legacy agreement. The actual cutover preserves old readout and segmented directories together under the activation's baseline directory. No old data are deleted. Windows share-mode probes and hashing occur before the short directory-rename sequence; handles must be released before renaming the parent. A partial rename failure rolls both directories back. This is not a universal remote/NFS lock, and target-process checks remain required.

`default_activation.json` records the active default receipt; current.json records default_entrypoint_active. Ordinary default consumers now read the activated data. `baseline_relocation.json` binds original paths and hashes to preserved files. Historical checkpoint validation resolves only an exact expected hash, never silently substitutes the new default as the old source. Prior molecular acceptance records are preserved and explicitly superseded. Legacy Sequence remains an assigned codebook word, not a recovered measurement; UnknownObservation maps to non-target Infeasible/P=0 for CSV compatibility.

## Qualified first-cycle pause extension

The default history mixture can leave systematic false negatives when a whole
image-derived region experienced a first-acquisition pause. Opt in with
`--qualify-first-pause`; `pause_mechanisms` compares conditional predictive evidence
on complementary observed-cycle folds, unique Q20 anchors, independent normal
controls, disjoint transcript-ID halves and coherent 1024px blocks. Both folds
require at least 100 affected and 100 normal anchors, affected pause preference
>=85% and normal <=15%; each half has at least 30 anchors and >=80%/<=20%
respectively. At least three blocks require >=30 anchors, >=85% preference and
halves of >=10 anchors at >=80%. Only a missing first cycle with all nine later
cycles observed is supported. Other cases retain the original mixture.

Inside qualified blocks, scores are conditional on the independently supported
pause. Within-history code collisions remain unknown; individual optical evidence
with Bayes factor >100 vetoes the regional assumption. The additional injected
first-pause qualification must pass the existing >=50 Q20 / <=1% disagreement
gate. Conditional confidence is not universal biological FDR.

For an already activated revision with identical observation validity and fitted
model, also pass `--localize-active`. The controller checks outside-policy
equivalence at absolute tolerance 1e-10, then emits only qualified-policy rows so
the current rows outside the policy remain exactly unchanged. Material differences
fail closed. A standalone `python -m sprintseq.observation_repair.localize` can
perform the same checked localization on an existing immutable candidate.

```powershell
sprintseq observation-repair --run-dir <processed-run> --output-root <new-generation> --model-dir <source-bound-model> --qualify-first-pause --localize-active
```

Inspect independent density/locality checks before publishing and activating a
successor. Use a new explicit `--activation-id` for every default-directory
transaction. Preserve old sources and receipts; do not rerun historical analysis
scripts that assume `readout/` still means the original baseline.

## Reaction history can invalidate the original detection window

A molecular repair that keeps the original point set cannot recover spots that
were never detected. `detection_window.detection_support` projects codewords
through the qualified history and counts measurable bright dimensions in the
original detection rounds. `supplement_cycles` proposes extra rounds from code
coverage, without using gene-expression abundance. With one initial pause, a
code whose first bright base was fourth becomes bright in acquisition five; a
rounds-1–4 detector has zero coverage for it. Audit every panel feature, not only
the genes used in a density figure.

Use the separate supervised recovery stage after first-pause qualification:

```powershell
python -m sprintseq.observation_repair.detection_pipeline --run-dir <processed-run> --policy-dir <qualified-pause-policy> --model-dir <source-bound-model> --output-root <new-generation> --control-offset-x <normal-region-offset-pixels>
```

It runs label-independent Spotiflow discovery on observed rounds 5–10 with the
original hybiss/global-normalization/0.01 recipe, reads all ten cycles at measured
positions, and decodes with the qualified history. Candidates use native 2px
deduplication and must be more than 3px from every existing point. Requiring two
*late detection rounds* is deliberately avoided: it creates another code-specific
blind spot. A source-pixel check instead requires a predicted-bright discovery
channel to have a local top-hat peak within the native radius-1 readout tolerance
and no brighter radius-3 neighbor. No candidate coordinates are interpolated or
shifted to smooth a density map.

The stage also processes a normal-region control (candidate-only), replays 256
old-spot intensity vectors exactly, and uses unchanged Q20 confidence. Native
minimum-intensity filtering is retained. `spatial_cells.observation_extension`
allocates new IDs and writes standard transcripts, intensities, assignments, legacy
inputs, matrices, density and fresh readout QC together. Every old row remains
unchanged. Existing assignment zeros are preserved; new assignment uses the
run's recorded threshold and mask. Default activation must preserve revised
position/intensity inputs instead of overwriting them with old static copies.

Normal-region additions measure the recall difference between detection windows;
they are not a false-positive truth set. Review source peaks, per-gene counts,
negative/ambiguous calls, and exact old-row preservation before publication.
Unresolved low-confidence genes remain flagged, not promoted by lowering Q20 or
borrowing neighboring gene counts.

## Review beyond a statistical HOLD

A gene-region flag is a review item, not proof of a technical artifact. Compare
independent morphology and cell-mask area before using a cell-rich annulus as a
reference for a background/edge region. Evaluate control-sector sensitivity and
overall callability; do not equalize a gene to pooled neighboring counts. Store
per-item dispositions and explicit unresolved measurements beside accepted output.
This adds context; it never reintroduces an outside-tissue/FOV expression filter.

History events strictly after the final measured cycle cannot affect an already
acquired prefix. `ObservableHistoryModel` removes those variables before the
qualified event-count limit and uses the same measured-dimensional likelihood,
class competition and collision handling. Events that affect later measured
cycles still count; no-evidence and unqualified relevant histories remain unknown.
Positive erasure controls and background activation controls qualify this path.

The default repair writer now refreshes assignment only for repaired IDs under
the unchanged mask and recorded assignment-probability threshold. Keeping a
pre-repair cell_id=0 after confidence recovery can otherwise suppress the cell
matrix despite a corrected density map. No threshold is invented when its
provenance is absent. Historical writers remain frozen for receipt replay.

Coordinate refinement is a separate source-bound recovery: independent observed
peaks must agree, avoid other old points, retain the original decoder threshold,
and pass position-offset controls. Update position/intensity sidecars and cell
assignment together; preserve every untouched CSV row exactly. The source-specific
CD1A pilot is not a general license to force weak hypotheses into a target gene.


## Default integration

The standard `sprintseq pipeline` / `experiments/run_pipeline.ps1` entry now runs automatic recovery, including optional C11 discovery and measured later-round fallback. Low-level commands below remain diagnostic building blocks. See [the standard automatic flow](methods/artifact_control.md#standard-automatic-recovery-entry) for output, restart and exception contracts.
