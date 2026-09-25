"""QC report orchestrators — tie metrics + plots together and write to disk."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .metrics import (
    compute_alert_flags,
    compute_gene_calling_metrics,
    compute_intensity_summary,
    compute_pipeline_funnel,
    compute_spatial_density_grid,
)
from .plots import plot_gene_calling_qc, plot_readout_qc


def _serialize(obj):
    """Make obj JSON-safe (handles numpy types)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _dump_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_serialize)


def generate_readout_qc(
    intensity_df: pd.DataFrame,
    position_df: pd.DataFrame,
    output_dir,
    run_id: str,
    channels: list,
    seq_cycle: int,
    detection_stats: dict,
    filter_stats: dict,
    dedup_stats: dict,
    img_shape=None,
    bin_size: int = 200,
    pixel_size: float = None,
) -> Path:
    """Generate readout QC JSON report + PNG figure."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    intensity_summary = compute_intensity_summary(intensity_df, channels, seq_cycle)
    funnel = compute_pipeline_funnel(
        detection_stats.get("total_raw", 0),
        filter_stats.get("n_after", 0),
        dedup_stats.get("n_after", 0),
    )

    spatial_grid = None
    if "Y" in position_df.columns and "X" in position_df.columns and len(position_df) > 0:
        y = position_df["Y"].to_numpy()
        x = position_df["X"].to_numpy()
        shape = img_shape or (int(y.max()) + 1, int(x.max()) + 1)
        spatial_grid = compute_spatial_density_grid(y, x, bin_size, shape)

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "detection": detection_stats,
        "filtering": filter_stats,
        "deduplication": dedup_stats,
        "intensity_summary": intensity_summary,
        "final_spot_count": len(position_df),
        "funnel": funnel,
    }

    json_path = output_dir / "readout_qc.json"
    _dump_json(report, json_path)

    fig = plot_readout_qc(intensity_summary, funnel, spatial_grid, channels, seq_cycle,
                          pixel_size=pixel_size, bin_size=bin_size)
    fig.savefig(output_dir / "readout_qc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return json_path


def generate_gene_calling_qc(
    result_df: pd.DataFrame,
    output_dir,
    run_id: str,
    diagnostics: dict = None,
) -> Path:
    """Generate gene-calling QC JSON report + PNG figure."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    diagnostics = diagnostics or {}

    metrics = compute_gene_calling_metrics(result_df)
    alerts = compute_alert_flags(metrics)
    metrics["alert_flags"] = alerts

    # Convergence info from PoSTcode diagnostics
    losses = diagnostics.get("losses", [])
    converged = False
    if len(losses) >= 10:
        last10 = losses[-10:]
        full_range = max(losses) - min(losses)
        converged = (max(last10) - min(last10)) < 0.01 * (full_range + 1e-10)
    metrics["convergence"] = {
        "n_iterations": len(losses),
        "final_loss": float(losses[-1]) if losses else None,
        "converged": converged,
        "losses": [float(v) for v in losses],
    }

    # Extra diagnostics
    for key in ("w_star", "n_training_spots", "n_total_spots"):
        if key in diagnostics:
            metrics[key] = diagnostics[key]

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    json_path = output_dir / "gene_calling_qc.json"
    _dump_json(report, json_path)

    prob = np.nan_to_num(result_df["Probability"].to_numpy(dtype=np.float64), nan=0.0) if len(result_df) > 0 else np.array([])
    entropy = np.nan_to_num(result_df["Entropy"].to_numpy(dtype=np.float64), nan=0.0) if len(result_df) > 0 else np.array([])
    fig = plot_gene_calling_qc(metrics, prob, entropy, result_df)
    fig.savefig(output_dir / "gene_calling_qc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return json_path
