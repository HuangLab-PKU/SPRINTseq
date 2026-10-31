# Observation-aware decoding for local imaging failures

Mandatory pre-readout prevention and all-panel acceptance are specified in [Standard artifact checks and recovery](artifact_control.md). The repair algorithms below are qualified recovery stages within that flow.

## Purpose and ownership

This method detects unreliable cycle/channel measurements and decodes each spot using its remaining observations, while retaining uncertainty about reaction history. It covers local attenuation, defocus and completed-acquisition missing stacks. It repairs molecular inference, not the missing image itself.

The method is first-party code, not a vendored external tool. Keep the image primitives in `spatial_img_core.observation_qc`, the sequential two-colour decoder and orchestration in `sprintseq.observation_repair`, and schema-compatible revision/activation in `spatial_cells`. These are already independent maintained packages; an additional Git submodule or `ext` copy would duplicate ownership. The SPRINTseq CLI is the user-facing integration point. It requires the other installed packages only when this method is run; ordinary SPRINTseq decoding does not acquire a hard downstream dependency.

The original image implementation in the experimental package is retained only for historical source-hash replay. New callers use the core package. This promotion preserves the segmentation algorithm and numerical thresholds; it is not a new biological validation.

## Inputs and scope

Use a finalized OME-TIFF, an exact run codebook, fitted PoSTcode model parameters, and transcript/intensity tables with matching immutable IDs and image coordinates. The integrated revision additionally requires accepted same-frame cell masks and assignments. The currently qualified integrated configuration is ten sequential cycles with two SBS channels, cy3 and cy5, evaluated on the existing 202-feature panel. The standalone segmentation API accepts at least five cycles and two channels; that broader array interface is not qualification of other sequencing protocols.

New acquisitions need independently reviewed normal and affected regions, acquisition layout checks, and the control tests below. Do not infer missing acquisitions from black pixels or use increased gene counts as a mechanism label.

## Image-derived validity

For each cycle/channel, compute local signal, local contrast, and high-frequency energy at a recorded image-pyramid level. Normalize global exposure differences by robust same-channel cross-cycle texture ratios. Compare the normalized local metrics with their cross-cycle medians. Attenuation and texture-loss criteria propose contiguous failure regions; closing, minimum component area, and an uncertain boundary band suppress isolated noise and expose edge uncertainty. These masks do not use decoded genes or Q20 counts.

The output is a reason mask per image plane and `observed[spot, cycle, channel]` with transcript IDs. Reason bits are attenuation 1, texture loss 2, source missing 4, and uncertain boundary 8. Optical mask sampling uses the documented pyramid-cell mapping. Renderer coverage uses exact level-0 pixel centres, `floor(x + 0.5)`, and row-packed little-endian support masks. A valid overlapping donor remains observed even when another tile is missing.

A separate metadata census compares expected cycle/channel/tile/Z observations with source files only after source-hashed acquisition completion evidence. Entire absent stacks become explicit missing declarations. Unfinished, empty or gapped stacks hold the pipeline. Registration must still establish observable geometry; this method does not invent transforms. Legacy runs without renderer support retain an explicit coverage-unavailable status.

## Molecular likelihood

Let x be a spot's transformed intensity vector, O its observed dimensions, g a latent class, and h a candidate acquisition history. Evaluate the marginal Gaussian likelihood using the corresponding mean subvector and covariance submatrix:

    L(g, h) = Normal(x[O]; mu[g, h][O], Sigma[O, O])
    P(g | x[O]) proportional to prior(g) * sum_h prior(h) * L(g, h)

All fitted target, background and infeasible classes participate. Missing channels are marginalized, never filled with zero: both-dark is a valid code symbol only when both channels were measured. Background means remain unchanged under reaction histories.

For eligible dual-channel image failures, compare unchanged history with up to two paused acquisitions. A pause consumes no base; later observations shift to the next unconsumed base. A four-state nuisance mixture represents an unobserved paused acquisition; terminal bases remain unobserved. Missing-source reason codes cannot generate chemical-pause hypotheses.

Run sensitivity checks with pause priors 0.1, 0.5 and 0.9. A retained label must agree across them; report probability, second probability and entropy from the least-confident case. Exact projected-code collisions across histories, insufficient observations and unsupported event counts become UnknownObservation with Q0. These probabilities are model-based confidence, not independently calibrated biological false-discovery rates.

## Distinguishing optical loss from reaction delay

Image darkness or blur alone cannot establish chemistry. Candidate regions come from image components. Within them, select identities from a unique high-confidence observed prefix; later observed cycles are held out for comparing unchanged and paused histories. Require unaffected normal controls, agreement between disjoint transcript-ID halves and spatially coherent local blocks.

The implemented v3 gates require at least 100 informative affected points, 30 controls, 99% prefix/reference agreement in controls, and pause preference no greater than 15% in the affected/control sets and both affected halves. At least three local blocks must pass their support/coherence gates. Only qualified blocks disable pause alternatives. Under this no-shift policy, first-cycle failures, pause-favoring regions, mixed mechanisms and insufficient support retain the history mixture. An opt-in positive first-cycle-pause policy now complements it, as described below. The policy binds exact transcript IDs/cycles to model and QC hashes and cannot expand outside eligible image failures.

Thus optical erasure and missing files share observed-dimensional decoding, but missing files have a known acquisition cause and never imply a reaction pause. Bubble-associated chemical delay remains a separately evaluated history hypothesis.

## Evaluation and acceptance

1. Validate image masks against independent reviewed regions, retaining pyramid resolution and legacy coverage limitations. Report normal-region false exclusions and missed artifacts where reference annotations exist; do not invent accuracy estimates without annotations.
2. Export the actual fitted model and verify full-observation numerical parity. The exporter fits a new reproducible model when historical parameters were not saved; it does not claim historical parameter recovery.
3. Use up to 8,192 unaffected high-confidence technical-reference spots. Test first-cycle pause, single-channel absence, dual-channel optical erasure, uncertain history and injected one/two pauses. Every condition needs at least 50 retained Q20 calls and no more than 1% disagreement. Report retention alongside disagreement. These controls establish implementation behavior under the fitted model, not biological truth.
4. Check mechanism locality and independent optical-only agreement where the no-pause policy applies. Preserve values outside affected IDs.
5. Compare before/after gene density at identical coordinates, scales and thresholds, including normal regions and suspicious gene-to-gene transitions. Smoother density or greater yield is supporting evidence, never an optimization target or proof of correctness.
6. Reconstruct cell-by-gene counts independently; validate unchanged mask assignments, every density plane, native/legacy reader agreement and output/source hashes. Publish only qualified generations. Explicit default activation is a separate transactional operation with rollback snapshots.

## Output contract and limitations

The method produces immutable validity, mechanism, control, repair and revision receipts; explicit unknown calls; standard transcript/assignment tables; count matrices; density products; and per-cell observed-coverage summaries. The revision writer currently retains the historical `cell_gene_matrix_Q20_panel202.csv` filename for consumer compatibility; columns are taken from the supplied codebook, not generated from that filename. A different panel still needs qualification. Count zeros remain observed counts, not proof of biological absence. Coverage derived from detected spots is not complete cell-pixel coverage.

It cannot recover spots absent from every detection cycle, synthesize terminal bases, certify arbitrary geometric distortions, or guarantee every changed label is closer to biological truth. Broad defects affecting most reference cycles can defeat cross-cycle segmentation. Different panels, cycle counts, chemistry or optics require renewed qualification.

## Use and reproducibility

See [the operational guide](../observation_repair.md) for commands and activation semantics. Run `sprintseq observation-repair --help`. Image-only detection and completed-acquisition census are available as `python -m spatial_img_core.observation_qc.points` and `python -m spatial_img_core.observation_qc.census`.

The standard `sprintseq pipeline` and PowerShell orchestrator automatically invoke qualified repair after a provisional decode and transcript export when image/panel audit requires it. Cell segmentation is optional: unsegmented runs receive verified molecular/density outputs; existing supported geometry is reused with refreshed assignments and matrices. Low-level `observation-repair` commands remain available for explicit forensic work. Already accepted defaults are source-checked and are not replayed. Run-specific evidence belongs in the canonical experiment record; keep patient/run tables, codebooks and model weights outside the library repository. This is an internal reusable method specification, not a claim of publication or algorithmic novelty.

## Positive first-cycle-pause policy

`pause_mechanisms` uses two complementary observed-cycle partitions (cycles 2–5
and 6–10). Under each history, it infers unique Q20 anchor identities on one
partition and evaluates the other by conditional predictive log density: joint
log evidence minus training-partition log evidence. Normal controls undergo the
same masking and scoring. Both complementary folds, both ID halves and local
blocks must pass the explicit gates in the operational guide. The image component
is proposed independently of those molecular scores.

A qualified region permits conditional decoding under a first-acquisition pause;
within-history ambiguity and a strong per-spot optical-evidence veto remain.
This addresses a limitation of unconditional cross-history collision rejection,
which can erase an entire gene in a region even when independent regional
evidence establishes the acquisition history. It does not justify imposing
a pause on an arbitrary circular density feature. First-cycle injected controls
and exact outside-policy preservation are required before a release.
