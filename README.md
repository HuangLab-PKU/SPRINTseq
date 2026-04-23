# sprintseq

Official analysis pipeline for **SPRINTseq** — *Spatially Resolved and signal-diluted Next-generation Targeted sequencing* — a rapid, signal-crowdedness-robust in situ sequencing method based on hybrid block coding (Chang et al., *PNAS* 2023).

This repository hosts the **post-stitched** side of the pipeline: spot detection, intensity readout, gene calling, density maps, and cell segmentation, all starting from already-stitched whole-slide images. The upstream imaging / stitching code lives in the sibling [`spatial-img-core`](https://github.com/HuangLab-PKU/spatial-img-core) package.

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
| `sprintseq.gene_calling` | Intensity correction (channel balance, decay, phasing) and gene mapping (postcode / threshold / intensity-direct / per-round-max). |
| `sprintseq.segment` | Cell segmentation (CellSAM / Cellpose with DAPI ± morphology channels) and RNA-to-cell assignment (mask-based or nucleus-centroid KD-tree). |
| `sprintseq.barcode_design` | Barcode graph design utilities (offline codebook generation). |

## Install

Designed for the `spatial-prep-dp` mamba env alongside `spatial-img-core` and `prism`. Python ≥ 3.10.

```powershell
mamba run -n spatial-prep-dp pip install -e <path-to-SPRINTseq>\code
```

### Optional deep-learning dependencies

Baseline install covers the classical DoG + tophat detection path and the threshold / intensity-direct gene-calling methods — no GPU required. For the DL stack (Spotiflow, PoSTcode, CellSAM, Cellpose), install per-backend:

| Backend | Install |
|---|---|
| Spotiflow (spot detection CNN) | `pip install sprintseq[detection-dl]` |
| Cellpose (segmentation) | `pip install sprintseq[cellpose]` |
| **PoSTcode** (probabilistic gene calling, *not on PyPI*) | `pip install -e <path-to-SPRINTseq>\experiments\src\postcode` |
| **CellSAM** (segmentation, *not on PyPI under this name*) | `pip install -e <path-to-SPRINTseq>\experiments\src\cellsam` |

Inside the HuangLab `spatial-prep-dp` env all four backends are usually already editable-installed from `experiments/src/`, so additional install steps are only needed on a fresh machine.

## CLI

After install a single `sprintseq` command is on `PATH`, dispatching four subcommands:

```powershell
sprintseq --help                                                              # subcommand list
sprintseq readout       --run-id <RUN_ID> [--detection-cycles 1-4,11] [--seq-cycles 10] [--channels cy3,cy5] [--n-workers 4]
sprintseq gene-calling  --run-id <RUN_ID> --ref-file <codebook.csv> [--seq-cycles 10] [--channels cy3,cy5]
sprintseq density       --run-id <RUN_ID> [--threshold 0.95] [--fac 200]
sprintseq segment       --run-id <RUN_ID> [--dapi <path>] [--morphology <file>]... [--model <name>] [--method {auto,cellsam,cellpose,nuclei-kdtree}]
```

Channel semantics: `--channels cy3,cy5` are the **SBS spot channels** — used for both detection and intensity readout. `--detection-cycles` accepts an explicit list (e.g. `11` for a total-spot staining cycle, `1-4,11` to union classic + total-spot). The `--dapi` / `--morphology` flags on `segment` are separate — morphology is any cell-body marker (FAM, CellMask, WGA, etc.), not required to be FAM. Large-image reads go through `tifffile.memmap` throughout (detection, intensity readout, segmentation prep), so only the active tile is paged into RAM even for 30k × 30k stitched images.

Each subcommand's `--help` prints a stage description, current defaults, and an example invocation.

Commands expect a processed-data directory at `\\10.10.10.1\NAS Processed Images\<RUN_ID>_processed\stitched\` with `cyc_{1..N}_{cy3,cy5}.tif` for SBS, and DAPI (± FAM / any cytoplasm or membrane channel) for segmentation.

For a PowerShell orchestrator that chains readout → gene-calling → density, see `../experiments/run_pipeline.ps1`. Segmentation is run separately because the chosen model and morphology channels are per-run decisions.

## Library use

```python
from sprintseq.readout import get_spot_coordinates, read_intensity_tophat, block_starts
from sprintseq.gene_calling import map_genes, correct_intensity, check_sequence
from sprintseq.segment import (
    prepare_cellsam_input, run_cellsam_segmentation,
    assign_spots_to_cells, assign_spots_to_nuclei_kdtree,
    auto_detect_dapi, auto_detect_morphology,
)
```

Heavy optional deps are lazy-imported: calling `get_spot_coordinates(..., method='spotiflow')`, `map_genes(..., method='postcode')`, or `run_cellsam_segmentation(...)` without the corresponding backend installed raises a clear `ImportError` with the install hint.
