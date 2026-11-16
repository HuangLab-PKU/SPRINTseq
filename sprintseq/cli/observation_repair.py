"""CLI adapter for source-bound observation repair; inference is imported lazily."""
from pathlib import Path


def build_parser(subparsers):
    parser = subparsers.add_parser(
        "observation-repair",
        help="Detect invalid observations and qualify a versioned molecular repair.",
        description="Image validity, mechanism qualification, masked decoding and downstream revision. "
                    "Requires existing transcripts and cell assignments. Publishing updates a versioned "
                    "pointer; default-directory activation is a separate operation.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--fit-device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--qualify-first-pause", action="store_true",
                        help="Qualify regional first-cycle pauses using held-out cycles and normal controls.")
    parser.add_argument("--localize-active", action="store_true",
                        help="Extend matching activated results only inside a qualified pause policy.")
    parser.set_defaults(_func=run)
    return parser


def run(args):
    from sprintseq.observation_repair.pipeline import run as run_pipeline
    return run_pipeline(args.run_dir, args.output_root, args.model_dir,
                        publish=args.publish, fit_device=args.fit_device, qc_dir=args.qc_dir,
                        qualify_first_pause=args.qualify_first_pause, localize_active=args.localize_active)
