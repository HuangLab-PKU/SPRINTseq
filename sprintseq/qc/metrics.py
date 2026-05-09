"""Pure-computation QC metrics for the SPRINTseq pipeline.

No I/O, no matplotlib — every function takes arrays/DataFrames in and returns
dicts/arrays out so they are trivial to unit-test.
"""

import numpy as np
import pandas as pd

MAX_QSCORE = 50.0


def compute_phred_qscore(probs: np.ndarray) -> np.ndarray:
    """Convert posterior probabilities to Phred Q-scores.

    Q = -10 * log10(1 - P), capped at MAX_QSCORE for P >= 1 - 10^(-MAX_QSCORE/10).
    """
    p = np.asarray(probs, dtype=np.float64)
    p_clamped = np.clip(p, 0.0, 1.0 - 10 ** (-MAX_QSCORE / 10))
    with np.errstate(divide="ignore"):
        q = -10.0 * np.log10(1.0 - p_clamped)
    q = np.clip(q, 0.0, MAX_QSCORE)
    return q


def compute_intensity_summary(
    intensity_df: pd.DataFrame, channels: list, seq_cycle: int
) -> dict:
    """Per-cycle per-channel intensity statistics."""
    result = {}
    for cyc in range(1, seq_cycle + 1):
        for ch in channels:
            col = f"cyc_{cyc}_{ch}"
            if col not in intensity_df.columns:
                continue
            vals = intensity_df[col].to_numpy(dtype=np.float64)
            p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
            result[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p5": float(p5),
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
                "p95": float(p95),
            }
    return result


def compute_pipeline_funnel(
    n_detected: int, n_filtered: int, n_deduped: int
) -> dict:
    """Readout pipeline funnel: detection -> filter -> dedup."""
    filter_drop = 1.0 - n_filtered / n_detected if n_detected > 0 else 0.0
    dedup_drop = 1.0 - n_deduped / n_filtered if n_filtered > 0 else 0.0
    return {
        "detection": n_detected,
        "after_filter": n_filtered,
        "after_dedup": n_deduped,
        "filter_drop_rate": filter_drop,
        "dedup_drop_rate": dedup_drop,
    }


def compute_spatial_density_grid(
    y: np.ndarray, x: np.ndarray, bin_size: int, img_shape: tuple
) -> np.ndarray:
    """2D histogram of spot positions, returning a (rows, cols) count grid."""
    h, w = img_shape
    n_rows = int(np.ceil(h / bin_size))
    n_cols = int(np.ceil(w / bin_size))
    y_edges = np.arange(0, n_rows + 1) * bin_size
    x_edges = np.arange(0, n_cols + 1) * bin_size
    grid, _, _ = np.histogram2d(
        np.asarray(y), np.asarray(x), bins=[y_edges, x_edges]
    )
    return grid.astype(np.int32)


def compute_gene_calling_metrics(result_df: pd.DataFrame) -> dict:
    """Compute comprehensive gene-calling QC metrics from a mapping result."""
    total = len(result_df)
    if total == 0:
        return {
            "total_spots": 0,
            "mapped_to_gene": 0,
            "background_count": 0,
            "infeasible_count": 0,
            "mapping_rate": 0.0,
            "background_rate": 0.0,
            "probability": {"mean": 0, "median": 0, "std": 0},
            "entropy": {"mean": 0, "median": 0, "std": 0},
            "q_score": {"mean": 0, "median": 0, "fraction_q20": 0, "fraction_q30": 0},
            "confidence_margin": {"mean": 0, "median": 0},
            "per_gene_stats": [],
            "alert_flags": [],
        }

    gene_col = result_df["Gene"]
    bkg_mask = gene_col == "Background"
    inf_mask = gene_col == "Infeasible"
    gene_mask = ~bkg_mask & ~inf_mask & gene_col.notna()

    bkg_count = int(bkg_mask.sum())
    inf_count = int(inf_mask.sum())
    mapped = int(gene_mask.sum())

    prob = result_df["Probability"].to_numpy(dtype=np.float64)
    entropy = result_df["Entropy"].to_numpy(dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0.0)
    entropy = np.nan_to_num(entropy, nan=0.0)
    q = compute_phred_qscore(prob)

    margin = np.zeros(total)
    if "Probability_2" in result_df.columns:
        margin = prob - result_df["Probability_2"].to_numpy(dtype=np.float64)

    # Per-gene stats (genes only, not bkg/inf)
    per_gene = []
    if mapped > 0:
        gene_df = result_df[gene_mask]
        for gene_name, grp in gene_df.groupby("Gene"):
            grp_prob = grp["Probability"].to_numpy(dtype=np.float64)
            per_gene.append({
                "gene": str(gene_name),
                "count": len(grp),
                "mean_prob": float(np.mean(grp_prob)),
                "mean_q": float(np.mean(compute_phred_qscore(grp_prob))),
            })
        per_gene.sort(key=lambda g: g["count"], reverse=True)

    metrics = {
        "total_spots": total,
        "mapped_to_gene": mapped,
        "background_count": bkg_count,
        "infeasible_count": inf_count,
        "mapping_rate": mapped / total,
        "background_rate": bkg_count / total,
        "probability": {
            "mean": float(np.mean(prob)),
            "median": float(np.median(prob)),
            "std": float(np.std(prob)),
        },
        "entropy": {
            "mean": float(np.mean(entropy)),
            "median": float(np.median(entropy)),
            "std": float(np.std(entropy)),
        },
        "q_score": {
            "mean": float(np.mean(q)),
            "median": float(np.median(q)),
            "fraction_q20": float(np.mean(q >= 20.0)),
            "fraction_q30": float(np.mean(q >= 30.0)),
        },
        "confidence_margin": {
            "mean": float(np.mean(margin)),
            "median": float(np.median(margin)),
        },
        "per_gene_stats": per_gene,
    }
    return metrics


def compute_alert_flags(metrics: dict) -> list:
    """Return Xenium-inspired alert flags based on QC metrics."""
    flags = []
    mr = metrics.get("mapping_rate", 1.0)
    if mr < 0.30:
        flags.append({
            "level": "error",
            "metric": "mapping_rate",
            "value": mr,
            "threshold": 0.30,
            "message": f"Mapping rate {mr:.1%} is critically low (<30%)",
        })
    elif mr < 0.50:
        flags.append({
            "level": "warning",
            "metric": "mapping_rate",
            "value": mr,
            "threshold": 0.50,
            "message": f"Mapping rate {mr:.1%} is below expected (50%)",
        })

    fq20 = metrics.get("q_score", {}).get("fraction_q20", 1.0)
    if fq20 < 0.50:
        flags.append({
            "level": "warning",
            "metric": "fraction_q20",
            "value": fq20,
            "threshold": 0.50,
            "message": f"Only {fq20:.1%} of spots reach Q>=20",
        })

    cm = metrics.get("confidence_margin", {}).get("mean", 1.0)
    if cm < 0.10:
        flags.append({
            "level": "warning",
            "metric": "confidence_margin_mean",
            "value": cm,
            "threshold": 0.10,
            "message": f"Mean confidence margin {cm:.3f} is low — assignments may be ambiguous",
        })

    br = metrics.get("background_rate", 0.0)
    if br > 0.30:
        flags.append({
            "level": "warning",
            "metric": "background_rate",
            "value": br,
            "threshold": 0.30,
            "message": f"Background rate {br:.1%} is high (>30%)",
        })

    return flags


def compute_density_metrics(
    df: pd.DataFrame,
    density_cube: np.ndarray,
    gene_names: np.ndarray,
    threshold: float,
    fac: int,
) -> dict:
    """QC metrics for the density stage."""
    total_density = density_cube.sum(axis=0)
    n_bins = total_density.size
    n_occupied = int((total_density > 0).sum())

    per_gene = []
    for i, gene in enumerate(gene_names):
        per_gene.append({"gene": str(gene), "count": int(density_cube[i].sum())})
    per_gene.sort(key=lambda g: g["count"], reverse=True)

    return {
        "total_spots": int(len(df)),
        "n_genes": int(len(gene_names)),
        "threshold": threshold,
        "fac": fac,
        "per_gene_counts": per_gene,
        "spatial_coverage": n_occupied / n_bins if n_bins > 0 else 0.0,
    }
