"""
Density map generation from gene mapping results.

Generates per-gene downsampled density TIFF images from postcode mapping.
Uses vectorized bincount for all genes in a single pass (~20s for 220 genes).

Usage:
    sprintseq density --run-id <run_id>
    sprintseq density --run-id <run_id> --threshold 0.8 --fac 100
"""

import os
import re
import time
import argparse
import logging

import numpy as np
import pandas as pd
from tifffile import TiffFile, imwrite
from tqdm import tqdm

from sprintseq.readout.mosaic import has_mosaic, list_mosaics, mosaic_shape

logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
DEFAULT_QUALITY = 20
DEFAULT_THRESHOLD = 0.99  # Q20
DEFAULT_FAC = 200
# Reference mosaics to try (in order) when auto-detecting the stitched canvas shape.
# (cycle, channel) rather than a filename: a converted run has no per-channel .tif.
REF_MOSAIC_CANDIDATES = [(11, 'DAPI'), (11, 'cy3'), (1, 'cy3')]


def get_stitched_shape(stc_dir):
    """Auto-detect the stitched canvas shape, on either storage backend.

    Tries REF_MOSAIC_CANDIDATES in order, then falls back to whatever mosaic exists --
    every channel shares the canvas, so any of them answers the question. Returns
    (height, width) without reading pixels.
    """
    for cyc, chn in REF_MOSAIC_CANDIDATES:
        if has_mosaic(stc_dir, cyc, chn):
            shape = mosaic_shape(stc_dir, cyc, chn)
            logger.info(f"Reference mosaic: cyc_{cyc}_{chn}, shape: {shape}")
            return shape
    available = list_mosaics(stc_dir)
    if available:
        cyc, chn = available[0]
        shape = mosaic_shape(stc_dir, cyc, chn)
        logger.info(f"Reference mosaic (fallback): cyc_{cyc}_{chn}, shape: {shape}")
        return shape
    raise FileNotFoundError(
        f"No stitched mosaic found in {stc_dir}. "
        f"Tried: {REF_MOSAIC_CANDIDATES}"
    )


def parse_gene_name(gene):
    """Remove SP_num_ or SPnum_ prefix from gene name.

    Examples:
        SP_356_AKT1_SNP -> AKT1_SNP
        SP1_CD3D -> CD3D
        CD3D -> CD3D
    """
    m = re.match(r"SP_?\d+_(.+)", gene)
    return m.group(1) if m else gene


def fill_missing_genes(density_dir, ref_file, target_shape):
    """Create zero-filled density TIFs for codebook genes absent from *density_dir*.

    Returns list of gene names that were filled.
    """
    codebook = pd.read_csv(ref_file)
    expected = sorted({parse_gene_name(g) for g in codebook['Gene']})
    existing = {os.path.splitext(f)[0] for f in os.listdir(density_dir) if f.endswith('.tif')}
    blank = np.zeros(target_shape, dtype=np.uint16)
    filled = []
    for gene in expected:
        if gene not in existing:
            imwrite(os.path.join(density_dir, f'{gene}.tif'), blank)
            filled.append(gene)
    if filled:
        logger.info("Filled %d missing genes with black: %s", len(filled), ', '.join(filled))
    return filled


def generate_density_maps(read_dir, stc_dir, density_dir, threshold=DEFAULT_THRESHOLD, fac=DEFAULT_FAC, ref_file=None):
    """Generate per-gene density TIFF maps using vectorized bincount.

    All genes are processed in a single pass over the coordinate data:
    1. Compute downsampled bin indices (y // fac, x // fac) for all spots
    2. Use np.bincount with combined (gene_id * grid_size + flat_idx) index
    3. Reshape result into (n_genes, target_rows, target_cols) cube
    4. Write each slice as a separate TIFF

    Parameters
    ----------
    read_dir : str
        Readout directory containing position.csv and mapping_postcode.csv
    stc_dir : str
        Stitched directory for auto-detecting image shape
    density_dir : str
        Output directory for density TIFFs
    threshold : float
        Minimum probability threshold for including a spot
    fac : int
        Downsample factor (block size for binning)
    ref_file : str, optional
        Codebook CSV. When provided, genes present in the codebook but absent
        from the data are written as all-zero (black) density TIFs.
    """
    t0 = time.time()

    # Auto-detect image shape
    ref_shape = get_stitched_shape(stc_dir)
    target_rows = (ref_shape[0] // fac) + 1
    target_cols = (ref_shape[1] // fac) + 1
    grid_size = target_rows * target_cols
    logger.info(f"Density grid: {target_rows} x {target_cols} (fac={fac})")

    # Load & merge
    logger.info("Loading position and mapping data...")
    position_df = pd.read_csv(os.path.join(read_dir, 'position.csv'), index_col=0)[['Y', 'X']]
    mapping_df = pd.read_csv(os.path.join(read_dir, 'mapping_postcode.csv'), index_col=0)[['Gene', 'Probability']]
    df = pd.merge(position_df, mapping_df, left_index=True, right_index=True)
    del position_df, mapping_df
    logger.info(f"Total spots: {len(df)}")

    # Filter
    df = df[~df['Gene'].isin(['Background', 'Infeasible'])]
    df['Gene'] = df['Gene'].map(parse_gene_name)
    df = df[df['Probability'] > threshold]
    logger.info(f"After filtering (threshold={threshold}): {len(df)} spots, {df['Gene'].nunique()} genes")

    if len(df) == 0:
        logger.warning("No spots after filtering. Exiting.")
        return

    # Vectorized bin computation
    y = df['Y'].values.astype(np.int64)
    x = df['X'].values.astype(np.int64)
    bin_y = np.clip(y // fac, 0, target_rows - 1)
    bin_x = np.clip(x // fac, 0, target_cols - 1)
    flat_idx = bin_y * target_cols + bin_x

    # Factorize genes
    gene_labels, gene_names = pd.factorize(df['Gene'].values)
    n_genes = len(gene_names)

    # Single-pass bincount for all genes
    logger.info(f"Computing density cube ({n_genes} genes)...")
    combined_idx = gene_labels.astype(np.int64) * grid_size + flat_idx
    counts = np.bincount(combined_idx, minlength=n_genes * grid_size)
    density_cube = counts.reshape(n_genes, target_rows, target_cols).astype(np.uint16)

    # Write TIFFs
    os.makedirs(density_dir, exist_ok=True)
    logger.info(f"Writing {n_genes} density maps to {density_dir}")
    for i, gene in enumerate(tqdm(gene_names, desc="Writing density maps")):
        imwrite(os.path.join(density_dir, f'{gene}.tif'), density_cube[i])

    if ref_file:
        fill_missing_genes(density_dir, ref_file, (target_rows, target_cols))

    elapsed = time.time() - t0
    logger.info(f"Done! {n_genes} density maps in {elapsed:.1f}s")

    return density_cube, gene_names, df


def run_pipeline(run_id, threshold=DEFAULT_THRESHOLD, fac=DEFAULT_FAC, ref_file=None, density_label=None):
    """Main entry point.

    Parameters
    ----------
    run_id : str
        Run identifier
    threshold : float
        Probability threshold
    fac : int
        Downsample factor
    ref_file : str, optional
        Codebook CSV; missing genes are filled with black density TIFs.
    density_label : str, optional
        Override for the density subdirectory suffix (e.g. ``'Q20'``).
        Defaults to ``str(threshold)``.
    """
    if density_label is None:
        density_label = str(threshold)
    dest_dir = os.path.join(BASE_DEST_DIRECTORY, f'{run_id}_processed')
    stc_dir = os.path.join(dest_dir, 'stitched')
    read_dir = os.path.join(dest_dir, 'readout')
    density_dir = os.path.join(read_dir, f'density_{density_label}')

    if not os.path.isdir(read_dir):
        raise FileNotFoundError(f"Readout directory not found: {read_dir}")

    logger.info("=" * 60)
    logger.info("Density Map Generation")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Threshold: {threshold}, Downsample factor: {fac}")
    if ref_file:
        logger.info(f"Codebook: {ref_file} (missing genes will be filled with black)")

    result = generate_density_maps(read_dir, stc_dir, density_dir, threshold=threshold, fac=fac, ref_file=ref_file)

    if result is not None:
        density_cube, gene_names, df_filtered = result
        try:
            from sprintseq.qc import generate_density_qc
            generate_density_qc(
                df_filtered=df_filtered,
                density_cube=density_cube,
                gene_names=gene_names,
                output_dir=read_dir,
                run_id=run_id,
                threshold=threshold,
                fac=fac,
                density_label=density_label,
            )
            logger.info(f"  density_{density_label}_qc.* saved to {read_dir}")
        except Exception as e:
            logger.warning(f"QC generation failed (non-fatal): {e}")


def main():
    """`python -m sprintseq.cli.density` fallback entry point. The canonical CLI is `sprintseq density` (see sprintseq.cli.main)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from sprintseq.cli import resolve_threshold_and_label

    parser = argparse.ArgumentParser(description='Generate per-gene density maps from postcode mapping')
    parser.add_argument('--run-id', type=str, required=True, help='Run ID to process')
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'Probability threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('-Q', '--quality', type=int, default=None,
                        help='Phred quality score (Q20=0.99, Q30=0.999). Overrides --threshold.')
    parser.add_argument('--fac', type=int, default=DEFAULT_FAC,
                        help=f'Downsample factor (default: {DEFAULT_FAC})')
    parser.add_argument('--ref-file', type=str, default=None,
                        help='Codebook CSV; missing genes filled with black density TIFs')
    args = parser.parse_args()

    quality = args.quality if args.quality is not None else (
        DEFAULT_QUALITY if args.threshold is None else None)
    prob, label = resolve_threshold_and_label(
        args.threshold or DEFAULT_THRESHOLD, quality)
    run_pipeline(args.run_id, threshold=prob, fac=args.fac, ref_file=args.ref_file, density_label=label)


if __name__ == "__main__":
    main()
