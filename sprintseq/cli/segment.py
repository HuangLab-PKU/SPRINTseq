"""
Cell segmentation + RNA-to-cell assignment for SPRINTseq.

Usage:
    sprintseq segment --run-id <run_id> [--morphology <file>]... [--model <name>] ...

Modes (selected via --method, default 'auto'):
  - cellsam         DAPI + morphology channels → CellSAM → cell masks → spot-to-cell assignment.
  - cellpose        DAPI + morphology channels → Cellpose → cell masks → spot-to-cell assignment.
  - nuclei-kdtree   DAPI only → cellSAM nucleus segmentation → centroids → KD-tree assignment
                    (each spot attributed to its nearest nucleus centroid).
  - auto            nuclei-kdtree if `--morphology` is empty AND no cyc_*_FAM.tif is
                    auto-discovered under stitched/; otherwise cellsam.

Output (under `<RUN_ID>_processed/segmented/`):
  - cellsam_mask.tif (or cellpose_mask.tif)
  - cell_positions.csv   (Cell_ID, X, Y, Area)
  - assigned_spots.csv   (X, Y, Gene, Probability, Cell_ID [, Distance for kdtree])
  - cell_gene_matrix.csv (Cell_ID × Gene counts)
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import tifffile

from sprintseq.segment import utils as seg

logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
DEFAULT_METHOD = 'auto'
DEFAULT_PROB = 0.8
DEFAULT_BLOCK_SIZE = 2048
DEFAULT_OVERLAP = 256
DEFAULT_KDTREE_MAX_DISTANCE = None   # None = no cap; tighten per tissue if needed


# ========== Pipeline ==========

def _resolve_inputs(stc_dir, dapi_arg, morphology_args):
    """Resolve DAPI + morphology paths from CLI args, auto-detecting where unspecified."""
    if dapi_arg:
        dapi = Path(dapi_arg)
        if not dapi.exists():
            raise FileNotFoundError(f"DAPI image not found: {dapi}")
    else:
        dapi = seg.auto_detect_dapi(stc_dir)
        logger.info(f"Auto-detected DAPI: {dapi}")

    if morphology_args:
        morphology = [Path(m) for m in morphology_args]
        missing = [m for m in morphology if not m.exists()]
        if missing:
            raise FileNotFoundError(f"Morphology image(s) not found: {missing}")
    else:
        # Auto-detect FAM first (the canonical default); empty list if absent.
        morphology = seg.auto_detect_morphology(stc_dir, names=("FAM",))
        if morphology:
            logger.info(f"Auto-detected morphology: {[str(m) for m in morphology]}")
        else:
            logger.info("No morphology image auto-detected; will fall back to nuclei-kdtree in auto mode.")

    return dapi, morphology


def _resolve_method(method_arg, morphology):
    if method_arg != 'auto':
        return method_arg
    return 'cellsam' if morphology else 'nuclei-kdtree'


def run_pipeline(run_id, dapi=None, morphology=None, method=DEFAULT_METHOD,
                 model=None, prob_threshold=DEFAULT_PROB,
                 block_size=DEFAULT_BLOCK_SIZE, overlap=DEFAULT_OVERLAP,
                 kdtree_max_distance=DEFAULT_KDTREE_MAX_DISTANCE):
    """Run cell segmentation + spot-to-cell assignment for one RUN_ID.

    See module docstring for mode semantics. Writes outputs under
    `<BASE>/<RUN_ID>_processed/segmented/`.
    """
    dest_dir = Path(BASE_DEST_DIRECTORY) / f"{run_id}_processed"
    stc_dir = dest_dir / 'stitched'
    read_dir = dest_dir / 'readout'
    seg_dir = dest_dir / 'segmented'

    if not stc_dir.is_dir():
        raise FileNotFoundError(f"Stitched directory not found: {stc_dir}")
    seg_dir.mkdir(parents=True, exist_ok=True)

    position_file = read_dir / 'position.csv'
    mapping_file = read_dir / 'mapping_postcode.csv'
    if not position_file.exists() or not mapping_file.exists():
        raise FileNotFoundError(
            f"Expected readout outputs missing. Run `sprintseq readout` and "
            f"`sprintseq gene-calling` first. Looked for: {position_file}, {mapping_file}"
        )

    logger.info("=" * 72)
    logger.info(f"SPRINTseq segmentation pipeline — RUN_ID: {run_id}")
    logger.info("=" * 72)

    # --- Inputs ---
    dapi_path, morphology_paths = _resolve_inputs(stc_dir, dapi, morphology)
    method = _resolve_method(method, morphology_paths)
    logger.info(f"Method: {method}")
    logger.info(f"DAPI: {dapi_path}")
    logger.info(f"Morphology ({len(morphology_paths)}): {[str(m) for m in morphology_paths]}")
    if method in ('cellsam', 'cellpose') and not model:
        logger.warning(
            "No --model specified for %s. Using backend default. "
            "This is usually wrong — the user has typically validated a specific "
            "model per tissue / panel. Check a prior run's run_rna_assignment.py for "
            "the chosen model name.", method
        )

    # --- Spots ---
    spots_df = seg.load_and_merge_spots(position_file, mapping_file,
                                        probability_threshold=prob_threshold)

    # --- Segmentation ---
    mask = None
    if method == 'cellsam':
        img = seg.prepare_cellsam_input(dapi_path, morphology_paths=morphology_paths)
        cellsam_kwargs = {}
        if model:
            cellsam_kwargs['model'] = model
        mask = seg.run_cellsam_segmentation(img, block_size=block_size, overlap=overlap,
                                            **cellsam_kwargs)
        mask_file = seg_dir / 'cellsam_mask.tif'

    elif method == 'cellpose':
        img = seg.prepare_cellpose_input(dapi_path, morphology_paths=morphology_paths)
        cellpose_kwargs = {}
        if model:
            cellpose_kwargs['pretrained_model'] = model
        mask = seg.run_cellpose_segmentation(img, **cellpose_kwargs)
        mask_file = seg_dir / 'cellpose_mask.tif'

    elif method == 'nuclei-kdtree':
        # Use cellSAM with DAPI-only input (empty morphology list) to get nuclei masks
        img = seg.prepare_cellsam_input(dapi_path, morphology_paths=None)
        cellsam_kwargs = {}
        if model:
            cellsam_kwargs['model'] = model
        mask = seg.run_cellsam_segmentation(img, block_size=block_size, overlap=overlap,
                                            **cellsam_kwargs)
        mask_file = seg_dir / 'nuclei_mask.tif'

    else:
        raise ValueError(f"Unknown method: {method}. "
                         "Choose from auto / cellsam / cellpose / nuclei-kdtree.")

    tifffile.imwrite(str(mask_file), mask)
    logger.info(f"Saved mask: {mask_file}")

    # --- Cell positions ---
    positions_df = seg.extract_cell_positions(mask)
    positions_file = seg_dir / 'cell_positions.csv'
    positions_df.to_csv(positions_file, index=False)
    logger.info(f"Saved cell positions: {positions_file} ({len(positions_df)} cells)")

    # --- Spot-to-cell assignment ---
    if method == 'nuclei-kdtree':
        assigned, matrix = seg.assign_spots_to_nuclei_kdtree(
            spots_df, positions_df, max_distance=kdtree_max_distance
        )
    else:
        assigned, matrix = seg.assign_spots_to_cells(spots_df, mask)

    assigned_file = seg_dir / 'assigned_spots.csv'
    matrix_file = seg_dir / 'cell_gene_matrix.csv'
    assigned.to_csv(assigned_file, index=False)
    matrix.to_csv(matrix_file)
    logger.info(f"Saved assignments: {assigned_file} ({len(assigned)} spots)")
    logger.info(f"Saved cell × gene matrix: {matrix_file} "
                f"({matrix.shape[0]} cells × {matrix.shape[1]} genes)")

    logger.info("Segmentation pipeline completed.")
    return mask, assigned, matrix


def main():
    """`python -m sprintseq.cli.segment` fallback entry point. The canonical CLI is `sprintseq segment` (see sprintseq.cli.main)."""
    parser = argparse.ArgumentParser(description='Cell segmentation + spot assignment for SPRINTseq.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dapi', default=None, help='DAPI image path (auto-detect if omitted).')
    parser.add_argument('--morphology', action='append', default=None,
                        help='Morphology (cyto/membrane) image path; repeat for multiple channels.')
    parser.add_argument('--method', default=DEFAULT_METHOD,
                        choices=['auto', 'cellsam', 'cellpose', 'nuclei-kdtree'])
    parser.add_argument('--model', default=None)
    parser.add_argument('--prob', type=float, default=DEFAULT_PROB)
    parser.add_argument('--block-size', type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP)
    parser.add_argument('--kdtree-max-distance', type=float, default=DEFAULT_KDTREE_MAX_DISTANCE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_pipeline(
        run_id=args.run_id, dapi=args.dapi, morphology=args.morphology,
        method=args.method, model=args.model, prob_threshold=args.prob,
        block_size=args.block_size, overlap=args.overlap,
        kdtree_max_distance=args.kdtree_max_distance,
    )


if __name__ == '__main__':
    main()
