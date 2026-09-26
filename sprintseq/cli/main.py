"""Unified CLI entry point for the sprintseq package.

Usage:
    sprintseq <subcommand> [options]

Subcommands:
    codebook        Fetch the run's decoding codebook from probe-bank.
    readout         Spot detection + intensity readout from stitched images.
    gene-calling    Intensity correction (optional) + gene mapping.

sprintseq stops at decoded spots. Density maps, segmentation, the cell x gene matrix, cell
maps and spot maps moved to the `spatial-cells` package (same flags): `sprintseq density`
etc. still run there when spatial-cells is installed, with a note to call it directly.

Run `sprintseq <subcommand> --help` for details on each.
"""
import argparse
import logging
import sys
from pathlib import Path

from sprintseq._moved import MOVED_COMMANDS, run_moved_command
from sprintseq.cli import codebook as codebook_mod
from sprintseq.cli import gene_calling as gc_mod
from sprintseq.cli import readout as readout_mod
from sprintseq.cli import parse_cycles, parse_channels


def _build_codebook(sub):
    p = sub.add_parser(
        "codebook",
        help="Fetch the run's decoding codebook from probe-bank and snapshot it.",
        description=codebook_mod.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id", type=str, required=True,
                   help="Run identifier; must be recorded in probe-bank with its pools.")
    p.add_argument("--bank", type=str, default=None,
                   help=f"probe-bank base URL (default: $SPRINTSEQ_PROBE_BANK or {codebook_mod.DEFAULT_BANK}).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write codebook.csv + codebook.json "
                        "(default: <RUN_ID>_processed/codebook/).")
    p.add_argument("--min-hamming", type=int, default=3,
                   help="Refuse a codebook with a codeword pair closer than this (default: 3).")
    p.add_argument("--force", action="store_true",
                   help="Replace an existing snapshot that differs.")
    p.set_defaults(_func=_run_codebook)


def _run_codebook(args):
    try:
        codebook_mod.run(args.run_id, bank=args.bank, out_dir=args.out_dir,
                         min_hamming=args.min_hamming, force=args.force)
    except codebook_mod.CodebookError as exc:
        logging.getLogger("sprintseq.codebook").error("%s", exc)
        sys.exit(1)


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
    p.add_argument(
        "--spotiflow-normalization", type=str, default=None,
        choices=list(readout_mod.SPOTIFLOW_NORMALIZATIONS),
        help=f"How Spotiflow input is scaled: 'block' = each block by its own p1/p99.8 "
             f"(Spotiflow's default), 'global' = one p1/p99.8 per mosaic shared by all "
             f"its blocks. (default: {readout_mod.SPOTIFLOW_NORMALIZATION!r})",
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
        spotiflow_normalization=args.spotiflow_normalization,
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
            "(classical Hamming<=1 match) and 'intensity_direct' "
            "(starfish MetricDistance-style similarity)."
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in MOVED_COMMANDS:
        return run_moved_command(argv[0], argv[1:])

    parser = argparse.ArgumentParser(
        prog="sprintseq",
        description=(
            "SPRINTseq post-stitched analysis pipeline, up to decoded spots.\n\n"
            "Run `sprintseq <subcommand> --help` for per-subcommand options. "
            "For the full pipeline, use the PowerShell orchestrator "
            "experiments/run_pipeline.ps1 -RunId <id> -RefFile <codebook.csv>."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  codebook        Fetch the run's codebook from probe-bank.\n"
            "  readout         Spot detection + intensity readout.\n"
            "  gene-calling    Gene mapping (postcode / threshold / ...).\n\n"
            "Moved to `spatial-cells` (same flags; still forwarded from here when it is "
            "installed):\n"
            "  " + ", ".join(MOVED_COMMANDS) + "\n\n"
            "Example end-to-end:\n"
            "  sprintseq codebook      --run-id <id>\n"
            "  sprintseq readout       --run-id <id>\n"
            "  sprintseq gene-calling  --run-id <id> --ref-file <id>_processed/codebook/codebook.csv\n"
            "  spatial-cells density   --run-id <id>\n"
            "  spatial-cells segment   --run-id <id> --model <validated-model>\n"
            "  spatial-cells cell-map  --run-id <id> --gene-file markers.txt\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="{codebook,readout,gene-calling}")

    _build_codebook(sub)
    _build_readout(sub)
    _build_gene_calling(sub)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Root logger: keep INFO + timestamp format for all subcommands
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args._func(args)


if __name__ == "__main__":
    main()
