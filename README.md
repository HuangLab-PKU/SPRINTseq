# sprintseq

Official analysis pipeline for **SPRINTseq** — *Spatially Resolved and signal-diluted Next-generation Targeted sequencing* — a rapid, signal-crowdedness-robust in situ sequencing method based on hybrid block coding (Chang et al., *PNAS* 2023).

This repository hosts the **post-stitched** side of the pipeline up to decoded spots: spot detection, intensity readout and gene calling, all starting from already-stitched whole-slide images. The upstream imaging / stitching code lives in the sibling [`spatial-img-core`](https://github.com/HuangLab-PKU/spatial-img-core) package.

**Downstream moved (2026-09-25).** Density maps, cell segmentation, the cell × gene matrix, per-cell gene maps and transcript point layers are platform-agnostic, so they moved to the `spatial-cells` package, shared with the lab's other imaging pipelines. Tag [`v0.1.0-pre-split`](https://github.com/HuangLab-PKU/SPRINTseq/tree/v0.1.0-pre-split) is the last version of this repository that carries them. sprintseq does not depend on spatial-cells; with it installed, the old `sprintseq density` / `segment` / `cell-map` / ... commands and imports are forwarded there with a note (see `sprintseq/_moved.py`).

## Citation

If you use SPRINTseq in your work, please cite:

> Chang et al. (2023). *Rapid and signal crowdedness-robust in situ sequencing through hybrid block coding.* Proceedings of the National Academy of Sciences, 120(47): e2309227120.
>
> - DOI: <https://doi.org/10.1073/pnas.2309227120>
> - PNAS: <https://www.pnas.org/doi/10.1073/pnas.2309227120>
> - PubMed: <https://pubmed.ncbi.nlm.nih.gov/37963245/>
> - PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10666108/>

## Repository

<https://github.com/HuangLab-PKU/SPRINTseq>

## Subpackages

| Subpackage | Purpose |
|---|---|
| `sprintseq.readout` | Block-based spot detection (Spotiflow / DoG + tophat) and intensity readout from stitched images. |
| `sprintseq.gene_calling` | Intensity correction (channel balance, decay, phasing) and gene mapping (postcode / threshold / intensity-direct). |
| `sprintseq.qc` | Readout and gene-calling QC reports. |
| `sprintseq.barcode_design` | Barcode graph design utilities (offline codebook generation). |

## Install

Designed for the lab's `spatial` mamba env alongside `spatial-img-core`, `prism` and `spatial-cells`. Python ≥ 3.10.

```powershell
mamba run -n spatial pip install -e <path-to-SPRINTseq>\code
```

### Optional deep-learning dependencies

Baseline install covers the classical DoG + tophat detection path and the threshold / intensity-direct gene-calling methods — no GPU required. For the DL stack, install per-backend:

| Backend | Install |
|---|---|
| Spotiflow (spot detection CNN) | `pip install sprintseq[detection-dl]` |
| **PoSTcode** (probabilistic gene calling, *not on PyPI*) | `pip install -e <path-to-SPRINTseq>\experiments\src\postcode` |

Inside the HuangLab `spatial` env both are already editable-installed from `experiments/src/`, so additional install steps are only needed on a fresh machine.

## CLI

After install a single `sprintseq` command is on `PATH`, dispatching these subcommands:

```powershell
sprintseq --help                                                              # subcommand list
sprintseq codebook      --run-id <RUN_ID> [--bank <url>] [--out-dir <dir>] [--min-hamming 3] [--force]
sprintseq readout       --run-id <RUN_ID> [--detection-cycles 1-4,11] [--seq-cycles 10] [--channels cy3,cy5] [--n-workers 4]
sprintseq gene-calling  --run-id <RUN_ID> --ref-file <RUN_ID>_processed/codebook/codebook.csv [--seq-cycles 10] [--channels cy3,cy5]
```

Then continue with `spatial-cells density | density-stack | segment | cell-map | spot-map --run-id <RUN_ID> ...` (same flags as the former sprintseq subcommands).

`codebook` asks probe-bank (`$SPRINTSEQ_PROBE_BANK`, default `http://10.10.10.1:8001`) for the run's decoding codebook -- derived from the pools the run records in the probe ledger (marker tube + the sample's TCR / allele tubes), labelled for analysis, with the run's decoys -- and snapshots it as `<RUN_ID>_processed/codebook/codebook.csv` (No., Gene, Barcode) plus `codebook.json` (pools, design versions, ledger commit, audit). The snapshot is the record of what the run was decoded with: a differing one is not replaced without `--force`, and an ambiguous codebook is refused. A run the ledger does not know yet has to be recorded there first (probe_design `bank/scripts/record_mix_guide_pools.py` + `record_run.py`).

Channel semantics: `--channels cy3,cy5` are the **SBS spot channels** — used for both detection and intensity readout. `--detection-cycles` accepts an explicit list (e.g. `11` for a total-spot staining cycle, `1-4,11` to union classic + total-spot). Large-image reads go through `tifffile.memmap` throughout (detection, intensity readout), so only the active tile is paged into RAM even for 30k × 30k stitched images.

Each subcommand's `--help` prints a stage description, current defaults, and an example invocation.

Commands expect a processed-data directory at `\\10.10.10.1\NAS Processed Images\<RUN_ID>_processed\stitched\` holding the stitched mosaic (`mosaic.ome.tif`, `mosaic.ome.zarr`, or legacy `cyc_{1..N}_{cy3,cy5}.tif`), read through `sprintseq.readout.mosaic`.

For a PowerShell orchestrator that chains readout → gene-calling, see `../experiments/run_pipeline.ps1`.

## Library use

```python
from sprintseq.readout import get_spot_coordinates, read_intensity_tophat, block_starts
from sprintseq.gene_calling import map_genes, correct_intensity, check_sequence
```

Heavy optional deps are lazy-imported: calling `get_spot_coordinates(..., method='spotiflow')` or `map_genes(..., method='postcode')` without the corresponding backend installed raises a clear `ImportError` with the install hint.
