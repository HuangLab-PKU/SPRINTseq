"""Transcript points for QuPath: the spot layer under the per-cell gene maps.

The cell maps answer "how much of gene G is in each cell"; this answers "where exactly are
the transcripts". Both are written in full-resolution mosaic pixels, so they overlay on
`stitched/mosaic.ome.tif` and on each other in QuPath.

Usage:
    sprintseq spot-map --run-id <run_id> --all
    sprintseq spot-map --run-id <run_id> --genes CD3E,KRT19 --per-spot
    sprintseq spot-map --run-id <run_id> --all --roi 20000:24000,30000:34000 --per-spot

Inputs (first that exists, unless --spots):
  - `<RUN_ID>_processed/segmented/assigned_spots.csv` (has Cell_ID), else
  - `<RUN_ID>_processed/readout/position.csv` + `mapping_<method>.csv`.

Output: `<that directory>/spots_<label>[_fovmasked][_roi][_perspot].geojson`.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd

from sprintseq.cli.cellmap import parse_roi, read_gene_file
from sprintseq.readout import spot_geojson as sg

logger = logging.getLogger(__name__)

BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
DEFAULT_QUALITY = 20
DEFAULT_THRESHOLD = 0.99          # Q20, as density and cell-map
DEFAULT_SPOTS = 'assigned_spots.csv'
DEFAULT_MAPPING = 'mapping_postcode.csv'
#: Above this many objects a per-spot layer makes QuPath crawl; the CLI refuses and says how
#: to narrow it (measured: the 285k-cell outline file already takes ~15 s to import).
PER_SPOT_LIMIT = 200_000


def resolve_spots(run_id, spots=None, base_dir=BASE_DEST_DIRECTORY, mapping=DEFAULT_MAPPING):
    """The spot table to draw, and the directory to write beside it.

    Prefers the segmented table (its ``Cell_ID`` is what makes a per-spot layer worth
    opening); falls back to merging `readout/position.csv` with the gene calls, so a run that
    has not been segmented can still be inspected.
    """
    proc = Path(base_dir) / f'{run_id}_processed'
    if spots:
        path = Path(spots)
        if not path.is_absolute():
            for parent in (proc / 'segmented', proc / 'readout', proc):
                if (parent / path).is_file():
                    path = parent / path
                    break
        if not path.is_file():
            raise FileNotFoundError(f"Spot table not found: {path}")
        return path, path.parent
    seg = proc / 'segmented' / DEFAULT_SPOTS
    if seg.is_file():
        return seg, seg.parent
    read_dir = proc / 'readout'
    pos, mp = read_dir / 'position.csv', read_dir / mapping
    if not (pos.is_file() and mp.is_file()):
        raise FileNotFoundError(
            f"No spot table: neither {seg} nor {pos} + {mp}. Run `sprintseq gene-calling` "
            "(and `sprintseq segment` for cell IDs) first.")
    return (pos, mp), read_dir


def merged_table(pos_path, map_path):
    """`position.csv` + `mapping_<method>.csv` joined on the spot index, as one frame."""
    position = pd.read_csv(pos_path, index_col=0)
    calls = pd.read_csv(map_path, index_col=0)
    keep = [c for c in ('Gene', 'Probability') if c in calls.columns]
    merged = position.join(calls[keep], how='inner')
    logger.info("Merged %s + %s -> %s spots", pos_path.name, map_path.name, f"{len(merged):,}")
    return merged


def run_pipeline(run_id, *, gene_file=None, genes=None, use_all=False,
                 threshold=DEFAULT_THRESHOLD, label=None, roi=None, spots=None,
                 exclude_fov_masked=False, per_spot=False, max_per_gene=None, force=False,
                 output=None, base_dir=BASE_DEST_DIRECTORY):
    """Write the transcript points of one RUN_ID as a QuPath GeoJSON. Returns a stats dict."""
    t0 = time.time()
    label = label or str(threshold)
    source, out_dir = resolve_spots(run_id, spots, base_dir)

    logger.info("=" * 60)
    logger.info("Spot map -- RUN_ID: %s", run_id)
    logger.info("=" * 60)
    wanted = None
    if gene_file:
        wanted = read_gene_file(gene_file)
    elif genes:
        wanted = [g.strip() for g in genes.split(',') if g.strip()]
    elif not use_all:
        raise ValueError("choose --all, --gene-file or --genes")
    logger.info("Source: %s", source if not isinstance(source, tuple)
                else f"{source[0].name} + {source[1].name}")

    if isinstance(source, tuple):
        table = merged_table(*source)
        tmp = out_dir / '.spot_map_merged.csv'          # load_spots reads a file, not a frame
        table.to_csv(tmp, index=False)
        try:
            spot_df = sg.load_spots(tmp, threshold=threshold, genes=wanted, roi=roi,
                                    exclude_fov_masked=exclude_fov_masked)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        spot_df = sg.load_spots(source, threshold=threshold, genes=wanted, roi=roi,
                                exclude_fov_masked=exclude_fov_masked)
    if spot_df.empty:
        raise ValueError("No spots left after filtering -- check the threshold, genes and ROI.")
    if wanted:
        absent = [g for g in wanted if g not in set(spot_df['Gene'])]
        if absent:
            logger.warning("%d requested gene(s) have no spot here: %s",
                           len(absent), ', '.join(absent))

    if per_spot and len(spot_df) > PER_SPOT_LIMIT and not force:
        raise ValueError(
            f"{len(spot_df):,} spots as individual QuPath objects would make panning crawl "
            f"(limit {PER_SPOT_LIMIT:,}). Narrow it with --genes / --roi / a higher -Q, drop "
            "--per-spot for the per-gene MultiPoint layer, or pass --force.")

    tag = f'spots_{label}' + ('_fovmasked' if exclude_fov_masked else '')
    roi_tag = f'_y{roi[0]}-{roi[1]}_x{roi[2]}-{roi[3]}' if roi else ''
    name = output or f'{tag}{roi_tag}{"_perspot" if per_spot else ""}.geojson'
    path = out_dir / name

    if per_spot:
        stats = sg.write_point_geojson(spot_df, path)
    else:
        stats = sg.write_multipoint_geojson(spot_df, path, max_per_gene=max_per_gene)
    stats['path'] = str(path)
    top = sorted(stats['per_gene'].items(), key=lambda kv: -kv[1])[:5]
    logger.info("Written: %s (%.1f MiB; %s spots of %d genes as %s; top: %s)", path,
                os.path.getsize(path) / 2**20, f"{stats['drawn']:,}", stats['genes'],
                "one detection each" if per_spot else "one MultiPoint per gene",
                ', '.join(f"{g} {n:,}" for g, n in top))
    if stats['genes_sampled']:
        logger.warning("%d gene(s) were sampled down to %s points each -- their density in the "
                       "layer is not the real one (each feature records the fraction)",
                       stats['genes_sampled'], f"{max_per_gene:,}")
    logger.info("Open the run's stitched/mosaic.ome.tif in QuPath, then drag this file onto "
                "it; the classification colours are per gene.")
    logger.info("Done in %.1fs", time.time() - t0)
    return stats


def add_arguments(p):
    """Flags shared by `sprintseq spot-map` and the module fallback entry point."""
    p.add_argument('--run-id', type=str, required=True, help='Run identifier.')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--all', dest='use_all', action='store_true',
                     help='Every gene in the spot table.')
    grp.add_argument('--gene-file', type=str,
                     help='Text file with one gene per line (as for density-stack / cell-map).')
    grp.add_argument('--genes', type=str, help='Comma-separated gene names.')
    p.add_argument('--threshold', type=float, default=None,
                   help='Minimum postcode Probability. Overrides -Q when set.')
    p.add_argument('-Q', '--quality', type=int, default=None,
                   help=f'Phred quality score (Q20=0.99, Q30=0.999). (default: Q{DEFAULT_QUALITY})')
    p.add_argument('--roi', type=parse_roi, default=None,
                   help="Restrict to 'y0:y1,x0:x1' in full-resolution mosaic pixels.")
    p.add_argument('--per-spot', action='store_true',
                   help='One QuPath detection per transcript, carrying its Probability and '
                        f'Cell ID, instead of one MultiPoint per gene. Refused above '
                        f'{PER_SPOT_LIMIT:,} spots without --force.')
    p.add_argument('--max-per-gene', type=int, default=None,
                   help='Sample a gene down to this many points (MultiPoint mode). The feature '
                        'records the fraction drawn. Default: draw every spot.')
    p.add_argument('--force', action='store_true',
                   help='Write a per-spot layer over the object limit anyway.')
    p.add_argument('--spots', type=str, default=None,
                   help=f'Spot table (absolute, or a name under segmented/ or readout/). '
                        f'Default: segmented/{DEFAULT_SPOTS}, else readout/position.csv + '
                        f'{DEFAULT_MAPPING}.')
    p.add_argument('--exclude-fov-masked', action='store_true',
                   help="Drop spots flagged 'fov_masked' (decode-side tile mask), as the "
                        'fovmasked cell × gene matrix and cell maps do.')
    p.add_argument('--output', type=str, default=None,
                   help='Output filename beside the spot table. Auto-named if omitted.')


def run_from_args(args):
    from sprintseq.cli import resolve_threshold_and_label
    quality = args.quality if args.quality is not None else (
        DEFAULT_QUALITY if args.threshold is None else None)
    prob, label = resolve_threshold_and_label(args.threshold or DEFAULT_THRESHOLD, quality)
    return run_pipeline(
        args.run_id, gene_file=args.gene_file, genes=args.genes, use_all=args.use_all,
        threshold=prob, label=label, roi=args.roi, spots=args.spots,
        exclude_fov_masked=args.exclude_fov_masked, per_spot=args.per_spot,
        max_per_gene=args.max_per_gene, force=args.force, output=args.output)
