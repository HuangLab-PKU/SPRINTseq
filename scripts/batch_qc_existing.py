"""Generate QC reports for existing processed runs (no pipeline re-run).

Reads position.csv, intensity.csv, mapping_postcode.csv from each run's
readout/ directory and produces readout_qc.*, gene_calling_qc.*, density_qc.*
alongside the existing files.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"\\10.10.10.1\NAS Processed Images")

RUN_IDS = [
    "20260430_ZCH_BZ23_mut_1_with_marker",
    "20260501_ZCH_BZ23_mut_1_with_marker_blocked",
    "20260501_ZCH_BZ23_mut_1_with_marker_blocked_lp",
    "20260504_ZCH_BZ23_mut_3_cDNA_with_ess_marker",
    "20260504_ZCH_BZ23_mut_3_cDNA_with_ess_marker_lp",
    "20260506_ZCH_BZ23_mut_4_iLock_all_marker_again",
    "20260506_ZCH_BZ23_mut_4_iLock_all_marker_again_lp",
]

CHANNELS = ["cy3", "cy5"]
SEQ_CYCLE = 10
DENSITY_THRESHOLD = 0.95
DENSITY_FAC = 200
PIXEL_SIZE = 0.1625  # um/pixel


def run_qc_for_run(run_id: str):
    dest = BASE / f"{run_id}_processed"
    read_dir = dest / "readout"
    stc_dir = dest / "stitched"
    density_dir = read_dir / f"density_{DENSITY_THRESHOLD}"

    print(f"\n{'='*70}")
    print(f"  {run_id}")
    print(f"{'='*70}")

    # --- Readout QC ---
    pos_path = read_dir / "position.csv"
    int_path = read_dir / "intensity.csv"
    if pos_path.exists() and int_path.exists():
        print("  [readout] Loading data...")
        position_df = pd.read_csv(pos_path)
        intensity_df = pd.read_csv(int_path)
        # Merge Y, X from position into intensity for QC
        merged = position_df[["index", "Y", "X"]].merge(
            intensity_df, on="index", how="inner"
        )

        # Reconstruct detection stats from position count
        n_final = len(position_df)
        detection_stats = {
            "per_channel": {},
            "total_raw": n_final,
            "total_after_exact_dedup": n_final,
        }
        filter_stats = {"n_before": n_final, "n_after": n_final, "threshold": 50}
        dedup_stats = {"n_before": n_final, "n_after": n_final}

        try:
            from sprintseq.qc import generate_readout_qc
            generate_readout_qc(
                intensity_df=merged,
                position_df=position_df,
                output_dir=read_dir,
                run_id=run_id,
                channels=CHANNELS,
                seq_cycle=SEQ_CYCLE,
                detection_stats=detection_stats,
                filter_stats=filter_stats,
                dedup_stats=dedup_stats,
                pixel_size=PIXEL_SIZE,
            )
            print(f"  [readout] OK -> {read_dir / 'readout_qc.json'}")
        except Exception:
            print(f"  [readout] FAILED")
            traceback.print_exc()
    else:
        print(f"  [readout] SKIP (missing CSV)")

    # --- Gene Calling QC ---
    map_path = read_dir / "mapping_postcode.csv"
    if map_path.exists():
        print("  [gene_calling] Loading mapping...")
        mapping_df = pd.read_csv(map_path)

        # Try to load convergence losses from log if available
        diagnostics = {}

        try:
            from sprintseq.qc import generate_gene_calling_qc
            generate_gene_calling_qc(
                result_df=mapping_df,
                output_dir=read_dir,
                run_id=run_id,
                diagnostics=diagnostics,
            )
            print(f"  [gene_calling] OK -> {read_dir / 'gene_calling_qc.json'}")
        except Exception:
            print(f"  [gene_calling] FAILED")
            traceback.print_exc()
    else:
        print(f"  [gene_calling] SKIP (no mapping_postcode.csv)")

    # --- Density QC (auto-discover all density_* dirs) ---
    density_dirs = sorted(read_dir.glob("density_*"))
    density_dirs = [d for d in density_dirs if d.is_dir()]
    if not density_dirs:
        print(f"  [density] SKIP (no density_* dirs)")
    for density_dir in density_dirs:
        label = density_dir.name.replace("density_", "")
        print(f"  [density:{label}] Loading data...")
        try:
            from tifffile import imread
            import re

            gene_tiffs = sorted(density_dir.glob("*.tif"))
            gene_names_arr = np.array([t.stem for t in gene_tiffs])
            if len(gene_tiffs) == 0:
                print(f"  [density:{label}] SKIP (no gene TIFFs)")
                continue

            first = imread(str(gene_tiffs[0]))
            cube = np.zeros((len(gene_tiffs), *first.shape), dtype=np.uint16)
            cube[0] = first
            for i, t in enumerate(gene_tiffs[1:], 1):
                cube[i] = imread(str(t))

            # Build a minimal filtered df from position + mapping
            pos_df = pd.read_csv(pos_path, index_col=0)[["Y", "X"]]
            map_df = pd.read_csv(map_path, index_col=0)[["Gene", "Probability"]]
            df = pd.merge(pos_df, map_df, left_index=True, right_index=True)
            df = df[~df["Gene"].isin(["Background", "Infeasible"])]
            def _parse_gene(g):
                m = re.match(r"SP_?\d+_(.+)", g)
                return m.group(1) if m else g
            df["Gene"] = df["Gene"].map(_parse_gene)
            df = df[df["Probability"] > DENSITY_THRESHOLD]

            from spatial_cells.qc import generate_density_qc   # density QC moved to spatial-cells
            generate_density_qc(
                df_filtered=df,
                density_cube=cube,
                gene_names=gene_names_arr,
                output_dir=str(read_dir),
                run_id=run_id,
                threshold=DENSITY_THRESHOLD,
                fac=DENSITY_FAC,
                density_label=label,
            )
            print(f"  [density:{label}] OK -> {read_dir / f'density_{label}_qc.json'}")
        except Exception:
            print(f"  [density:{label}] FAILED")
            traceback.print_exc()


if __name__ == "__main__":
    for rid in RUN_IDS:
        run_qc_for_run(rid)
    print(f"\n{'='*70}")
    print("Done!")
