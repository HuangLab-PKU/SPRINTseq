"""Per-cell gene maps: segmented cells painted with their transcript counts.

The cell-level counterpart of `sprintseq density` + `sprintseq density-stack`.
Every cell's mask footprint is filled with that cell's count for a gene, one
channel per gene, so the map shows real cell positions and shapes and the
viewer's pixel readout is the per-cell count. See `sprintseq.segment.cellmap`.

Usage:
    sprintseq cell-map --run-id <run_id> --gene-file genes.txt
    sprintseq cell-map --run-id <run_id> --all --exclude-fov-masked
    sprintseq cell-map --run-id <run_id> --gene-file genes.txt --format imagej

Inputs (under `<RUN_ID>_processed/segmented/`, from `sprintseq segment`):
  - the label mask: cellsam_mask.tif, else cellpose_mask.tif, else nuclei_mask.tif
  - assigned_spots.csv (Y, X, Gene, Probability, Cell_ID[, fov_masked])

Outputs (same directory):
  --format ome (default): cellmap_<label>_<gene-file stem | all>.ome.tif
      pyramidal OME-TIFF, channels [total, genes...]; QuPath / Fiji (Bio-Formats) load
      only the tiles and resolution level on screen.
  --format imagej: cellmap_<label>_<stem>.zip + cellmap_<label>_total.zip
      ImageJ-ZIP, opens in plain ImageJ with thermal LUT + display range, loaded whole.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd

from sprintseq.cli.density import parse_gene_name
from sprintseq.segment import cellmap as cm

logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
DEFAULT_QUALITY = 20
DEFAULT_THRESHOLD = 0.99  # Q20, as density
DEFAULT_FORMAT = 'ome'
# Full-resolution map pixel per format. OME: 0.65 um/px, a median cell ~11 x 11 px, cheap
# because viewers read tiles on demand. ImageJ: 1.625 um/px (~4 x 4 px) so a full panel
# stays under the 4 GiB an ImageJ TIFF can hold and fits in memory.
DEFAULT_FAC = {'ome': 4, 'imagej': 10}
DEFAULT_PIXEL_SIZE_UM = 0.1625
DEFAULT_SPOTS = 'assigned_spots.csv'
MASK_CANDIDATES = ('cellsam_mask.tif', 'cellpose_mask.tif', 'nuclei_mask.tif')


def parse_roi(value):
    """Parse 'y0:y1,x0:x1' (full-resolution pixels, half-open) into a 4-tuple."""
    if value is None:
        return None
    try:
        ys, xs = value.split(',')
        y0, y1 = (int(v) for v in ys.split(':'))
        x0, x1 = (int(v) for v in xs.split(':'))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid ROI {value!r}; expected 'y0:y1,x0:x1' in full-resolution pixels.")
    if y1 <= y0 or x1 <= x0:
        raise argparse.ArgumentTypeError(f"Empty ROI {value!r}: need y0 < y1 and x0 < x1.")
    return y0, y1, x0, x1


def resolve_mask(seg_dir, mask=None):
    """Explicit --mask (absolute or relative to segmented/), else the first candidate present."""
    if mask:
        path = Path(mask) if Path(mask).is_absolute() else Path(seg_dir) / mask
        if not path.is_file():
            raise FileNotFoundError(f"Mask not found: {path}")
        return path
    found = [Path(seg_dir) / m for m in MASK_CANDIDATES if (Path(seg_dir) / m).is_file()]
    if not found:
        raise FileNotFoundError(f"No label mask in {seg_dir}; looked for {MASK_CANDIDATES}. "
                                f"Run `sprintseq segment` first or pass --mask.")
    if len(found) > 1:
        logger.info("Several masks present (%s); using %s. Pass --mask to choose another.",
                    ', '.join(p.name for p in found), found[0].name)
    return found[0]


def read_gene_file(gene_file):
    genes = []
    with open(gene_file, encoding='utf-8') as fh:
        for line in fh:
            g = parse_gene_name(line.strip())
            if g and g not in genes:
                genes.append(g)
    if not genes:
        raise ValueError(f"No gene names in {gene_file}")
    return genes


def load_spots(spots_path, threshold, exclude_fov_masked):
    header = pd.read_csv(spots_path, nrows=0).columns
    usecols = [c for c in cm.SPOT_COLUMNS + ['fov_masked'] if c in header]
    raw = pd.read_csv(spots_path, usecols=usecols)
    spots = cm.filter_spots(raw, threshold, exclude_fov_masked=exclude_fov_masked)
    floor = raw.loc[~raw['Gene'].isin(cm.NON_GENE), 'Probability'].min()
    if pd.notna(floor) and floor - threshold > 1e-4:
        logger.warning("%s was already cut stricter than P > %.4g (lowest spot P = %.6g); "
                       "a looser threshold cannot add spots back.", spots_path.name, threshold, floor)
    logger.info("Spots: %s rows in %s -> %s kept (P > %.4g%s)", f"{len(raw):,}", spots_path.name,
                f"{len(spots):,}", threshold, ", fov-masked spots dropped" if exclude_fov_masked else "")
    return spots


def run_pipeline(run_id, *, gene_file=None, use_all=False, threshold=DEFAULT_THRESHOLD,
                 label=None, fmt=DEFAULT_FORMAT, fac=None, roi=None, mask=None, spots=None,
                 exclude_fov_masked=False, display_max=None,
                 min_border_area=cm.DEFAULT_MIN_BORDER_AREA,
                 pixel_size_um=DEFAULT_PIXEL_SIZE_UM, output=None,
                 base_dir=BASE_DEST_DIRECTORY):
    """Build the per-cell gene maps for one RUN_ID (see module docstring for outputs)."""
    t0 = time.time()
    label = label or str(threshold)
    fac = fac or DEFAULT_FAC[fmt]
    seg_dir = Path(base_dir) / f'{run_id}_processed' / 'segmented'
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}. "
                                f"Run `sprintseq segment` first.")
    mask_path = resolve_mask(seg_dir, mask)
    spots_path = Path(spots) if spots else seg_dir / DEFAULT_SPOTS
    if not spots_path.is_absolute():
        spots_path = seg_dir / spots_path
    if not spots_path.is_file():
        raise FileNotFoundError(f"Spot table not found: {spots_path}")

    tag = f'cellmap_{label}' + ('_fovmasked' if exclude_fov_masked else '')
    roi_tag = f'_y{roi[0]}-{roi[1]}_x{roi[2]}-{roi[3]}' if roi else ''
    stem = 'all' if use_all else Path(gene_file).stem
    suffix = '.ome.tif' if fmt == 'ome' else '.zip'
    out_path = seg_dir / (output or f'{tag}_{stem}{roi_tag}{suffix}')
    total_path = seg_dir / f'{tag}_total{roi_tag}.zip' if fmt == 'imagej' else None

    logger.info("=" * 60)
    logger.info("Cell map builder -- RUN_ID: %s", run_id)
    logger.info("=" * 60)
    logger.info("Mask: %s", mask_path)
    logger.info("Format: %s; full resolution fac=%d -> %.4g um/px%s", fmt, fac, pixel_size_um * fac,
                f", ROI y {roi[0]}:{roi[1]} x {roi[2]}:{roi[3]}" if roi else "")

    spot_df = load_spots(spots_path, threshold, exclude_fov_masked)
    genes = sorted(spot_df['Gene'].unique()) if use_all else read_gene_file(gene_file)
    logger.info("Genes: %d from %s", len(genes), 'the spot table' if use_all else gene_file)

    stats = cm.build_cell_maps(
        mask_path, spot_df, genes, out_path, fmt=fmt, total_path=total_path,
        fac=fac, px_um=pixel_size_um * fac, roi=roi,
        min_border_area=min_border_area, display_max=display_max,
    )

    logger.info("Mask/spot Cell_ID agreement: %.2f%% of %s spots checked",
                100 * stats['match_fraction'], f"{stats['match_checked']:,}")
    logger.info("Map %d x %d px; %s cells drawn, %s cells with spots centred in view "
                "(%d restored at centroid, %d lost to a taken centroid pixel)",
                stats['map_shape'][0], stats['map_shape'][1], f"{stats['n_cells_shown']:,}",
                f"{stats['n_cells_in_view']:,}", stats['n_restored'], stats['n_lost'])
    if stats['genes_without_spots']:
        logger.warning("%d gene(s) have no spots and are all-zero slices: %s",
                       len(stats['genes_without_spots']), ', '.join(stats['genes_without_spots']))
    if fmt == 'imagej':
        for path, raw, rng in ((out_path, stats['stack_bytes'], stats['display_max']),
                               (total_path, stats['total_bytes'], stats['total_display_max'])):
            logger.info("Written: %s (%.1f MiB; %.0f MiB in ImageJ; display 0-%g)",
                        path, os.path.getsize(path) / 2**20, raw / 2**20, rng)
    else:
        logger.info("Written: %s (%.1f MiB; %d channels [total + genes], %d levels %s; "
                    "%.1f GiB if uncompressed)", out_path, stats['file_bytes'] / 2**20,
                    len(genes) + 1, len(stats['level_shapes']),
                    ' > '.join(f'{w}x{h}' for h, w in stats['level_shapes']),
                    stats['raw_bytes'] / 2**30)
    logger.info("Done in %.1fs", time.time() - t0)
    return stats


def add_arguments(p):
    """Flags shared by `sprintseq cell-map` and the module fallback entry point."""
    p.add_argument('--run-id', type=str, required=True, help='Run identifier.')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--gene-file', type=str,
                     help='Text file with one gene per line (slice order = file order), '
                          'e.g. the gene file behind a density stack.')
    grp.add_argument('--all', dest='use_all', action='store_true',
                     help='One slice per gene in the spot table, alphabetical.')
    p.add_argument('--threshold', type=float, default=None,
                   help='Minimum postcode Probability for a spot to count. Overrides -Q when set.')
    p.add_argument('-Q', '--quality', type=int, default=None,
                   help=f'Phred quality score (Q20=0.99, Q30=0.999). (default: Q{DEFAULT_QUALITY})')
    p.add_argument('--format', dest='fmt', choices=['ome', 'imagej'], default=DEFAULT_FORMAT,
                   help="'ome': pyramidal OME-TIFF for QuPath / Fiji Bio-Formats, read tile by "
                        "tile. 'imagej': ImageJ-ZIP for plain ImageJ, thermal LUT preset, "
                        f"loaded whole. (default: {DEFAULT_FORMAT})")
    p.add_argument('--fac', type=int, default=None,
                   help='Downsample factor from the stitched mosaic for the full-resolution '
                        'map; one map pixel = fac x fac mosaic pixels. (default: '
                        + ', '.join(f'{k} {v} = {v * DEFAULT_PIXEL_SIZE_UM:.4g} um/px'
                                    for k, v in DEFAULT_FAC.items()) + ')')
    p.add_argument('--roi', type=parse_roi, default=None,
                   help="Restrict to 'y0:y1,x0:x1' in full-resolution mosaic pixels; "
                        "pair with a small --fac for a close-up.")
    p.add_argument('--mask', type=str, default=None,
                   help=f'Label mask (absolute or relative to segmented/). '
                        f'Default: first of {", ".join(MASK_CANDIDATES)}.')
    p.add_argument('--spots', type=str, default=None,
                   help=f'Spot table with Cell_ID (absolute or relative to segmented/). '
                        f'(default: {DEFAULT_SPOTS})')
    p.add_argument('--exclude-fov-masked', action='store_true',
                   help="Drop spots flagged 'fov_masked' (as cell_gene_matrix_fovmasked.csv).")
    p.add_argument('--display-max', type=float, default=None,
                   help='(imagej format) Display maximum for the gene stack. '
                        '(default: 99th percentile of non-zero per-cell counts)')
    p.add_argument('--min-border-area', type=int, default=cm.DEFAULT_MIN_BORDER_AREA,
                   help=f'Cells of at least this many map pixels get a 1-px border where they '
                        f'touch a neighbour; 0 disables borders. '
                        f'(default: {cm.DEFAULT_MIN_BORDER_AREA})')
    p.add_argument('--pixel-size', type=float, default=DEFAULT_PIXEL_SIZE_UM,
                   help=f'Mosaic pixel size in um, for the image calibration. '
                        f'(default: {DEFAULT_PIXEL_SIZE_UM})')
    p.add_argument('--output', type=str, default=None,
                   help='Output filename in segmented/ (ome: .ome.tif; imagej: .zip = '
                        'ImageJ-ZIP, .tif = plain). Auto-named if omitted.')


def run_from_args(args):
    from sprintseq.cli import resolve_threshold_and_label
    quality = args.quality if args.quality is not None else (
        DEFAULT_QUALITY if args.threshold is None else None)
    prob, label = resolve_threshold_and_label(args.threshold or DEFAULT_THRESHOLD, quality)
    return run_pipeline(
        args.run_id, gene_file=args.gene_file, use_all=args.use_all,
        threshold=prob, label=label, fmt=args.fmt, fac=args.fac, roi=args.roi,
        mask=args.mask, spots=args.spots, exclude_fov_masked=args.exclude_fov_masked,
        display_max=args.display_max, min_border_area=args.min_border_area,
        pixel_size_um=args.pixel_size, output=args.output,
    )


def main():
    """`python -m sprintseq.cli.cellmap` fallback entry point. The canonical CLI is `sprintseq cell-map`."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Per-cell gene maps (cells painted with counts).')
    add_arguments(parser)
    run_from_args(parser.parse_args())


if __name__ == '__main__':
    main()
