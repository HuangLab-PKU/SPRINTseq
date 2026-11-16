# Processed run output layout

New runs keep scientific products and resumable execution evidence separate:

- `stitched/`: final image products.
- `readout/`: `intensity.csv`, `position.csv`, method-specific `mapping_*.csv`, readout and gene-calling QC figures/JSON, required model metadata, one `density_Q20/` directory and `density_Q20_all.tif` with the standard Thermal LUT.
- `segmented/`: accepted masks, cell positions, spot assignments and cell-expression matrices.
- `logs/`: runtime logs (`readout/readout.log`), stage receipts, observation checks, failed-registration diagnostics (`registration_history/registration_unresolved.json`).
- `archive/`: superseded products, fitting intermediates and `observation_repair/`; historical pointer transactions live under `archive/result_history/`.

Acquisition/codebook inputs, solved registration transforms and current published-result pointers remain authoritative inputs at their established locations. A successful process or a tidy folder does not replace source-bound scientific acceptance.

Legacy checkpoints may be read in place. Do not rename a live workspace or rewrite hashed provenance to force this layout. Both old and new observation workspaces existing simultaneously are an error. New outputs never create root observation checks/repair/history directories. Explicit caller-selected custom outputs remain caller-owned.

PR31 and PR46 use `logs/prism31/` and `logs/prism46/` for stage records and gene counts, with fitting tables under `archive/prism31_intermediates/` and `archive/prism46_intermediates/`. VIS count summaries and locks use `logs/vis/`. PR46 emits `mapping_gmm.csv`; legacy `spots_called.csv` is a read compatibility path only.
