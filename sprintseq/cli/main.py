"""Unified CLI entry point for the sprintseq package.

Usage:
    sprintseq <subcommand> [options]

Subcommands:
    readout         Spot detection + intensity readout from stitched images.
    gene-calling    Intensity correction (optional) + gene mapping.
    density         Per-gene downsampled density TIFFs.
    segment         Cell segmentation + RNA-to-cell assignment.
    cell-map        Per-cell gene maps: segmented cells painted with their counts.
    spot-map        Transcript points for QuPath (one MultiPoint per gene, or per spot).

Run `sprintseq <subcommand> --help` for details on each.
"""
import argparse
import logging
import sys
from pathlib import Path

# Re-use module-level defaults and run_pipeline helpers from each CLI module.
# These imports are lazy where possible to keep --help fast and avoid pulling
# in heavy deps before the user chooses a subcommand.
from sprintseq.cli import cellmap as cellmap_mod
from sprintseq.cli import density as density_mod
from sprintseq.cli import density_stack as ds_mod
from sprintseq.cli import gene_calling as gc_mod
from sprintseq.cli import readout as readout_mod
from sprintseq.cli import segment as segment_mod
from sprintseq.cli import spotmap as spotmap_mod
from sprintseq.cli import parse_cycles, parse_channels, resolve_threshold_and_label


def _build_readout(sub):
    p = sub.add_parser(
        "readout",
        help="Spot detection + intensity readout from stitched images.",
        description=(
            "Stage 1 of the SPRINTseq post-stitched pipeline.\n\n"
            "Reads cyc_{1..N}_{cy3,cy5}.tif from "
            r"\\10.10.10.1\NAS Processed Images\<RUN_ID>_processed\stitched\, "
            "runs block-parallel spot detection (Spotiflow by default, DoG+tophat fallback), "
            "extracts tophat-corrected intensities, deduplicates across block overlaps, and writes "
            "readout/position.csv + readout/intensity.csv.\n\n"
            "Defaults (override via flags — edit sprintseq/cli/readout.py to change globally): "
            f"DETECTION_CYCLES={readout_mod.DETECTION_CYCLES}, "
            f"SEQ_CYCLE={readout_mod.SEQ_CYCLE} sequencing cycles (always consecutive 1..N), "
            f"CHANNELS={readout_mod.CHANNELS}, "
            f"SNRS={readout_mod.SNRS}, "
            f"DETECTION_METHOD={readout_mod.DETECTION_METHOD!r}, "
            f"BLOCK_SIZE={readout_mod.BLOCK_SIZE}, BLOCK_OVERLAP={readout_mod.BLOCK_OVERLAP}, "
            f"MIN_INTENSITY_THRESHOLD={readout_mod.MIN_INTENSITY_THRESHOLD}, "
            f"DEDUPLICATE_THRESHOLD={readout_mod.DEDUPLICATE_THRESHOLD}."
        ),
        epilog=(
            "Examples:\n"
            "  # Classic: detect on cyc_1..4\n"
            "  sprintseq readout --run-id <id>\n\n"
            "  # Total-spot protocol: one dedicated cycle stains every spot\n"
            "  sprintseq readout --run-id <id> --detection-cycles 11\n\n"
            "  # Union both strategies for max recall; also override SBS channels\n"
            "  sprintseq readout --run-id <id> --detection-cycles 1-4,11 --channels cy3,cy5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id", type=str, default=readout_mod.DEFAULT_RUN_ID,
        help=f"Run identifier; used to locate <base>/<RUN_ID>_processed/stitched/. "
             f"(default: {readout_mod.DEFAULT_RUN_ID})",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help=f"Process-pool size for block-parallel detection and intensity readout. "
             f"(default: {readout_mod.N_WORKERS})",
    )
    p.add_argument(
        "--detection-cycles", type=parse_cycles, default=None,
        help=f"Explicit list of cycles to detect spots in. Accepts comma-separated "
             f"integers and ranges, e.g. '1,2,3,4', '11', '1-4,11'. "
             f"(default: {readout_mod.DETECTION_CYCLES})",
    )
    p.add_argument(
        "--seq-cycles", type=int, default=None,
        help=f"Number of sequencing cycles (consecutive 1..N) read for intensity. "
             f"(default: {readout_mod.SEQ_CYCLE})",
    )
    p.add_argument(
        "--channels", type=parse_channels, default=None,
        help=f"Comma-separated SBS spot channels, used for BOTH detection and intensity. "
             f"(default: {','.join(readout_mod.CHANNELS)})",
    )
    p.add_argument(
        "--detection-method", type=str, default=None,
        choices=list(readout_mod.DETECTION_METHODS),
        help=f"Spot detection method dispatched in get_spot_coordinates. "
             f"'spotiflow' (DL), 'blob_log' (skimage scale-space LoG, "
             f"approximates Fiji TrackMate LogDetector), or one of the classical "
             f"feature-extraction methods backed by find_maxima. "
             f"(default: {readout_mod.DETECTION_METHOD!r})",
    )
    p.set_defaults(_func=_run_readout)


def _run_readout(args):
    # File logging to readout.log in the run's readout dir
    read_dir = Path(readout_mod.BASE_DEST_DIRECTORY) / f"{args.run_id}_processed" / "readout"
    read_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(read_dir / "readout.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(readout_mod._LOG_FMT))
    logging.getLogger().addHandler(fh)
    readout_mod.run_pipeline(
        args.run_id,
        n_workers=args.n_workers,
        detection_cycles=args.detection_cycles,
        seq_cycle=args.seq_cycles,
        channels=args.channels,
        detection_method=args.detection_method,
    )


def _build_gene_calling(sub):
    p = sub.add_parser(
        "gene-calling",
        help="Intensity correction (optional) + gene mapping.",
        description=(
            "Stage 2 of the SPRINTseq post-stitched pipeline.\n\n"
            "Reads readout/position.csv + readout/intensity.csv, applies global intensity "
            "correction (channel balance, signal decay, phasing) only if non-postcode methods "
            "are enabled (postcode handles these internally), then runs gene mapping against "
            "the supplied codebook. Writes readout/mapping_<method>.csv, "
            "readout/global_correction_info.json, readout/convergence_<method>.png, and "
            "readout/mapping_qc_<method>.png.\n\n"
            "Default method is 'postcode' (requires the vendored postcode package + torch + pyro). "
            "Alternatives documented in sprintseq/cli/gene_calling.py include 'threshold' "
            "(classical Hamming<=1 match) and 'intensity_direct' / 'per_round_max' "
            "(starfish-style similarity)."
        ),
        epilog="Example: sprintseq gene-calling --run-id 20260420_... --ref-file codebook/ZCH_TNBC_marker_genes.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id", type=str, required=True,
        help="Run identifier; reads readout/position.csv + readout/intensity.csv under this run's processed dir.",
    )
    p.add_argument(
        "--ref-file", type=str, required=True,
        help="Path to codebook CSV (columns: Barcode, Gene). Gene field may contain '+'-separated plex entries.",
    )
    p.add_argument(
        "--seq-cycles", type=int, default=None,
        help=f"Number of sequencing cycles to decode (matches readout's --seq-cycles). "
             f"(default: {gc_mod.SEQ_CYCLE})",
    )
    p.add_argument(
        "--channels", type=parse_channels, default=None,
        help=f"Comma-separated SBS channels (matches readout's --channels). "
             f"(default: {','.join(gc_mod.CHANNELS)})",
    )
    p.set_defaults(_func=_run_gene_calling)


def _run_gene_calling(args):
    gc_mod.run_pipeline(
        run_id=args.run_id, ref_file=args.ref_file,
        seq_cycle=args.seq_cycles, channels=args.channels,
    )


def _build_density(sub):
    p = sub.add_parser(
        "density",
        help="Per-gene downsampled density TIFFs from postcode mapping.",
        description=(
            "Stage 3 of the SPRINTseq post-stitched pipeline.\n\n"
            "Reads readout/mapping_postcode.csv and readout/position.csv, filters by "
            "probability >= threshold, uses a vectorised bincount to produce a "
            "per-gene cube, and writes one <gene>.tif per gene into readout/density/. "
            "Output resolution is (H/fac, W/fac)."
        ),
        epilog="Example: sprintseq density --run-id 20260420_... --threshold 0.9 --fac 100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id", type=str, required=True,
        help="Run identifier.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Minimum postcode Probability for a spot to be counted. "
             "Overrides -Q when set explicitly.",
    )
    p.add_argument(
        "-Q", "--quality", type=int, default=None,
        help=f"Phred quality score (Q20=0.99, Q30=0.999). "
             f"(default: Q{density_mod.DEFAULT_QUALITY})",
    )
    p.add_argument(
        "--fac", type=int, default=density_mod.DEFAULT_FAC,
        help=f"Downsample factor; output resolution is (H/fac, W/fac). "
             f"(default: {density_mod.DEFAULT_FAC})",
    )
    p.add_argument(
        "--ref-file", type=str, default=None,
        help="Codebook CSV; genes in the codebook but absent from mapping data "
             "are written as all-black (zero) density TIFs.",
    )
    p.set_defaults(_func=_run_density)


def _run_density(args):
    quality = args.quality if args.quality is not None else (
        density_mod.DEFAULT_QUALITY if args.threshold is None else None)
    prob, label = resolve_threshold_and_label(
        args.threshold or density_mod.DEFAULT_THRESHOLD, quality)
    density_mod.run_pipeline(args.run_id, threshold=prob, fac=args.fac,
                             ref_file=args.ref_file, density_label=label)


def _build_density_stack(sub):
    p = sub.add_parser(
        "density-stack",
        help="Composite density stack TIFF with thermal LUT.",
        description=(
            "Post-processing step: reads per-gene density TIFFs from "
            "readout/density_<threshold>/, applies Gaussian blur, and writes a "
            "single multi-page TIFF with embedded thermal colormap and display range.\n\n"
            "The output opens in ImageJ with the correct LUT and brightness already set."
        ),
        epilog=(
            "Examples:\n"
            "  # Stack genes from a text file:\n"
            "  sprintseq density-stack --run-id 20260402_... --gene-file marker_genes.txt\n\n"
            "  # Stack all density TIFs:\n"
            "  sprintseq density-stack --run-id 20260402_... --all\n\n"
            "  # Custom blur and display range:\n"
            "  sprintseq density-stack --run-id 20260402_... --all --sigma 1.0 --display-max 20"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id", type=str, required=True,
        help="Run identifier.",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--gene-file", type=str,
        help="Text file with one gene name per line.",
    )
    grp.add_argument(
        "--all", dest="use_all", action="store_true",
        help="Use all density TIFs in the directory.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help=f"Density subdirectory threshold suffix. (default: {ds_mod.DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "-Q", "--quality", type=int, default=None,
        help="Phred quality score; reads from density_Q<N>/ directory.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output filename (written to readout/). Auto-generated if omitted.",
    )
    p.add_argument(
        "--sigma", type=float, default=ds_mod.DEFAULT_SIGMA,
        help=f"Gaussian blur sigma. (default: {ds_mod.DEFAULT_SIGMA})",
    )
    p.add_argument(
        "--display-min", type=float, default=ds_mod.DEFAULT_DISPLAY_MIN,
        help=f"ImageJ display range minimum. (default: {ds_mod.DEFAULT_DISPLAY_MIN})",
    )
    p.add_argument(
        "--display-max", type=float, default=ds_mod.DEFAULT_DISPLAY_MAX,
        help=f"ImageJ display range maximum. (default: {ds_mod.DEFAULT_DISPLAY_MAX})",
    )
    p.add_argument(
        "--sort", action="store_true",
        help="Sort genes alphabetically. Default: preserve gene-file / codebook order.",
    )
    p.set_defaults(_func=_run_density_stack)


def _run_density_stack(args):
    quality = args.quality if args.quality is not None else (
        ds_mod.DEFAULT_QUALITY if args.threshold is None else None)
    _, label = resolve_threshold_and_label(
        args.threshold or ds_mod.DEFAULT_THRESHOLD, quality)
    ds_mod.run_pipeline(
        args.run_id, density_label=label,
        gene_file=args.gene_file, use_all=args.use_all, output=args.output,
        sigma=args.sigma, display_min=args.display_min, display_max=args.display_max,
        sort=args.sort,
    )


def _build_segment(sub):
    p = sub.add_parser(
        "segment",
        help="Cell segmentation + RNA-to-cell assignment.",
        description=(
            "Stage 4 of the SPRINTseq pipeline (run after gene-calling).\n\n"
            "Loads high-confidence spots from readout/mapping_postcode.csv, segments cells from "
            "stitched DAPI + optional morphology channels, and writes "
            "segmented/{mask.tif, cell_positions.csv, assigned_spots.csv, cell_gene_matrix.csv}.\n\n"
            "Methods:\n"
            "  auto           (default) nuclei-kdtree if no morphology is given AND none is auto-\n"
            "                 detected under stitched/; cellsam otherwise.\n"
            "  cellsam        DAPI + morphology channels → CellSAM → cell masks → spot-to-cell by mask.\n"
            "  cellpose       DAPI + morphology channels → Cellpose → cell masks → spot-to-cell by mask.\n"
            "  nuclei-kdtree  DAPI only → cellSAM nucleus segmentation → centroids → KD-tree:\n"
            "                 each spot attributed to its nearest nucleus centroid.\n\n"
            "IMPORTANT: the CellSAM / Cellpose model is a user-validated choice per tissue and panel. "
            "If --model is omitted, the backend default is used and a warning is logged — prefer to pass "
            "the model name your lab has validated for this panel."
        ),
        epilog=(
            "Examples:\n"
            "  # Auto-detect everything (FAM under stitched/ → cellsam; else → nuclei-kdtree):\n"
            "  sprintseq segment --run-id 20260420_ZCH_BZ29_Ca_TNBC_marker_4 --model cellsam_general\n\n"
            "  # Explicit DAPI + two morphology channels (e.g. FAM + WGA):\n"
            "  sprintseq segment --run-id <id> --method cellsam --model my_finetuned_v3 \\\n"
            "      --dapi stitched/cyc_11_DAPI.tif \\\n"
            "      --morphology stitched/cyc_11_FAM.tif --morphology stitched/cyc_11_WGA.tif\n\n"
            "  # Force nucleus-only + KD-tree (no morphology available):\n"
            "  sprintseq segment --run-id <id> --method nuclei-kdtree --kdtree-max-distance 40"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id", type=str, required=True,
        help="Run identifier; reads stitched/ and readout/ under this run's processed dir.",
    )
    p.add_argument(
        "--dapi", type=str, default=None,
        help="DAPI image path. Auto-detected from stitched/cyc_{11,1}_DAPI.tif if omitted.",
    )
    p.add_argument(
        "--morphology", action="append", default=None,
        help="Morphology (cytoplasm / membrane / cell-body) channel image. "
             "Repeat for multiple channels (they get merged via pixel-wise max). "
             "If omitted, tries to auto-detect cyc_*_FAM.tif; "
             "if still empty AND --method=auto, falls back to nuclei-kdtree.",
    )
    p.add_argument(
        "--method", default=segment_mod.DEFAULT_METHOD,
        choices=["auto", "cellsam", "cellpose", "nuclei-kdtree"],
        help=f"Segmentation method. (default: {segment_mod.DEFAULT_METHOD})",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Backend model name. Strongly recommended — do not rely on defaults "
             "unless that's the one your lab has validated for this panel.",
    )
    p.add_argument(
        "--prob", type=float, default=segment_mod.DEFAULT_PROB,
        help=f"Postcode probability threshold for including a spot. "
             f"(default: {segment_mod.DEFAULT_PROB})",
    )
    p.add_argument(
        "--block-size", type=int, default=segment_mod.DEFAULT_BLOCK_SIZE,
        help=f"Tile size for block-parallel segmentation. (default: {segment_mod.DEFAULT_BLOCK_SIZE})",
    )
    p.add_argument(
        "--overlap", type=int, default=segment_mod.DEFAULT_OVERLAP,
        help=f"Tile overlap in pixels. (default: {segment_mod.DEFAULT_OVERLAP})",
    )
    p.add_argument(
        "--kdtree-max-distance", type=float, default=segment_mod.DEFAULT_KDTREE_MAX_DISTANCE,
        help="(nuclei-kdtree only) Pixel radius beyond which a spot is considered unassigned. "
             "None = no cap.",
    )
    p.set_defaults(_func=_run_segment)


def _run_segment(args):
    segment_mod.run_pipeline(
        run_id=args.run_id, dapi=args.dapi, morphology=args.morphology,
        method=args.method, model=args.model, prob_threshold=args.prob,
        block_size=args.block_size, overlap=args.overlap,
        kdtree_max_distance=args.kdtree_max_distance,
    )


def _build_cell_map(sub):
    p = sub.add_parser(
        "cell-map",
        help="Per-cell gene maps: segmented cells painted with their counts.",
        description=(
            "Post-processing step after `segment` -- the cell-level counterpart of "
            "density + density-stack.\n\n"
            "Downsamples the segmentation mask by --fac, fills every cell's footprint with its "
            "count for each gene (postcode P > threshold, the same cut as density), and writes "
            "the maps to segmented/. Touching cells get a 1-px black border; cells smaller than "
            "one map pixel are kept as a pixel at their centroid. The pixel value is the "
            "per-cell count.\n\n"
            "--format ome (default): one pyramidal OME-TIFF, channels [total, genes...], tiled + "
            "deflate, 2x levels painted from the mask. QuPath and Fiji's Bio-Formats importer "
            "read only the tiles / level on screen (BZ07, 110 genes at 0.65 um/px: 98 MB file, "
            "~0.3 GB heap in Fiji as a virtual stack). Fiji shows gene names as slice labels once "
            "the Bio-Formats slice label pattern contains %w.\n\n"
            "--format imagej: ImageJ-ZIP stack + separate total map for plain ImageJ, thermal "
            "LUT and display range preset, loaded whole into memory."
        ),
        epilog=(
            "Examples:\n"
            "  # Same gene file as a density stack -> segmented/cellmap_Q20_<stem>.ome.tif\n"
            "  sprintseq cell-map --run-id <id> --gene-file markers.txt\n\n"
            "  # Every gene, spots in decode-masked tiles dropped\n"
            "  sprintseq cell-map --run-id <id> --all --exclude-fov-masked\n\n"
            "  # Plain-ImageJ zip (thermal LUT preset) at 1.625 um/px\n"
            "  sprintseq cell-map --run-id <id> --gene-file markers.txt --format imagej\n\n"
            "  # One Fiji macro line, once, to label Bio-Formats slices with channel (gene) names:\n"
            "  call(\"ij.Prefs.set\", \"bioformats.sliceLabelPattern\", \"%c%w\");"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cellmap_mod.add_arguments(p)
    p.set_defaults(_func=cellmap_mod.run_from_args)


def _build_spot_map(sub):
    p = sub.add_parser(
        "spot-map",
        help="Transcript points for QuPath (one MultiPoint per gene, or per spot).",
        description=(
            "The spot layer under the per-cell gene maps: the decoded transcripts as QuPath "
            "objects, in full-resolution mosaic pixels, so they overlay stitched/mosaic.ome.tif "
            "and the cell outlines from cell-map.\n\n"
            "Default: ONE annotation per gene holding every spot of that gene as a MultiPoint "
            "-- ~10^2 objects for a whole section, so QuPath pans at full speed, and each gene "
            "is toggled through its classification (colour is derived from the gene name, so it "
            "is the same in every run).\n\n"
            "--per-spot: one detection per transcript carrying its Probability and Cell ID, for "
            "a crop or a couple of genes. A million individual objects make QuPath crawl, so "
            "this is refused above the object limit unless you narrow it (--genes / --roi / a "
            "higher -Q) or pass --force.\n\n"
            "Reads segmented/assigned_spots.csv when the run is segmented (that is where Cell ID "
            "comes from), else readout/position.csv + mapping_postcode.csv."
        ),
        epilog=(
            "Examples:\n"
            "  # whole section, every gene -> segmented/spots_Q20.geojson\n"
            "  sprintseq spot-map --run-id <id> --all\n\n"
            "  # decode-masked tiles dropped, as in the fovmasked matrices\n"
            "  sprintseq spot-map --run-id <id> --all --exclude-fov-masked\n\n"
            "  # two genes, every transcript individually selectable\n"
            "  sprintseq spot-map --run-id <id> --genes CD3E,KRT19 --per-spot\n\n"
            "  # a 4000 x 4000 px crop, per spot\n"
            "  sprintseq spot-map --run-id <id> --all --roi 20000:24000,30000:34000 --per-spot"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    spotmap_mod.add_arguments(p)
    p.set_defaults(_func=spotmap_mod.run_from_args)


def main():
    parser = argparse.ArgumentParser(
        prog="sprintseq",
        description=(
            "SPRINTseq post-stitched analysis pipeline.\n\n"
            "Run `sprintseq <subcommand> --help` for per-subcommand options. "
            "For the full pipeline, use the PowerShell orchestrator "
            "experiments/run_pipeline.ps1 -RunId <id> -RefFile <codebook.csv>."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  readout         Spot detection + intensity readout.\n"
            "  gene-calling    Gene mapping (postcode / threshold / ...).\n"
            "  density         Per-gene density TIFFs.\n"
            "  density-stack   Composite density stack with thermal LUT.\n"
            "  segment         Cell segmentation + RNA-to-cell assignment.\n"
            "  cell-map        Per-cell gene maps (cells painted with counts).\n"
            "  spot-map        Transcript points for QuPath.\n\n"
            "Example end-to-end:\n"
            "  sprintseq readout       --run-id <id>\n"
            "  sprintseq gene-calling  --run-id <id> --ref-file <codebook.csv>\n"
            "  sprintseq density       --run-id <id>\n"
            "  sprintseq density-stack --run-id <id> --gene-file markers.txt\n"
            "  sprintseq segment       --run-id <id> --model <validated-model>\n"
            "  sprintseq cell-map      --run-id <id> --gene-file markers.txt\n"
            "  sprintseq spot-map      --run-id <id> --all\n"
        ),
    )
    sub = parser.add_subparsers(dest="command",
                                metavar="{readout,gene-calling,density,density-stack,segment,"
                                        "cell-map,spot-map}")

    _build_readout(sub)
    _build_gene_calling(sub)
    _build_density(sub)
    _build_density_stack(sub)
    _build_segment(sub)
    _build_cell_map(sub)
    _build_spot_map(sub)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Root logger: keep INFO + timestamp format for all subcommands
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args._func(args)


if __name__ == "__main__":
    main()
