"""QC figure generation for the SPRINTseq pipeline.

Each function returns a matplotlib Figure object; callers save to disk.
"""

import numpy as np
import pandas as pd

from .metrics import compute_phred_qscore


def plot_readout_qc(
    intensity_summary: dict,
    funnel: dict,
    spatial_grid: np.ndarray,
    channels: list,
    seq_cycle: int,
    pixel_size: float = None,
    bin_size: int = 200,
):
    """2x2 readout QC composite figure.

    Panels:
      1. Intensity heatmap (cycles x channels, median values)
      2. Grouped violin plot: x=cycle, hue=channel (shows decay + balance + distribution)
      3. Spatial spot density heatmap (real coordinates in um if pixel_size given)
      4. Pipeline funnel bar chart
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Readout QC", fontsize=14)

    n_cyc = seq_cycle
    n_ch = len(channels)

    # --- Panel 1: intensity heatmap ---
    ax = axes[0, 0]
    heatmap = np.zeros((n_cyc, n_ch))
    for ci, cyc in enumerate(range(1, n_cyc + 1)):
        for chi, ch in enumerate(channels):
            key = f"cyc_{cyc}_{ch}"
            heatmap[ci, chi] = intensity_summary.get(key, {}).get("p50", 0)
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(channels)
    ax.set_yticks(range(n_cyc))
    ax.set_yticklabels([f"cyc_{i}" for i in range(1, n_cyc + 1)], fontsize=7)
    ax.set_title("Median intensity per cycle/channel")
    for ci in range(n_cyc):
        for chi in range(n_ch):
            ax.text(chi, ci, f"{heatmap[ci, chi]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # --- Panel 2: grouped violin/box showing distribution per cycle per channel ---
    ax = axes[0, 1]
    ch_colors = {"cy3": "#2ca02c", "cy5": "#d62728"}
    fallback_colors = plt.cm.tab10.colors
    positions_all = []
    data_all = []
    colors_all = []
    tick_positions = []
    tick_labels = []
    group_width = n_ch + 1
    for ci, cyc in enumerate(range(1, n_cyc + 1)):
        tick_positions.append(ci * group_width + (n_ch - 1) / 2)
        tick_labels.append(f"cyc_{cyc}")
        for chi, ch in enumerate(channels):
            key = f"cyc_{cyc}_{ch}"
            stats = intensity_summary.get(key, {})
            box_stats = {
                "med": stats.get("p50", 0),
                "q1": stats.get("p25", 0),
                "q3": stats.get("p75", 0),
                "whislo": stats.get("p5", 0),
                "whishi": stats.get("p95", 0),
            }
            pos = ci * group_width + chi
            positions_all.append(pos)
            data_all.append(box_stats)
            colors_all.append(ch_colors.get(ch, fallback_colors[chi % len(fallback_colors)]))

    bp = ax.bxp(
        [{"med": d["med"], "q1": d["q1"], "q3": d["q3"],
          "whislo": d["whislo"], "whishi": d["whishi"], "fliers": []}
         for d in data_all],
        positions=positions_all,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Intensity")
    ax.set_title("Intensity distribution per cycle/channel (p5–p95 box)")
    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=ch_colors.get(ch, "gray"), alpha=0.6, label=ch) for ch in channels]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: spatial density (real coordinates in um) ---
    ax = axes[1, 0]
    if spatial_grid is not None and spatial_grid.size > 0:
        if pixel_size is not None:
            bin_um = bin_size * pixel_size
            rows, cols = spatial_grid.shape
            extent = [0, cols * bin_um, rows * bin_um, 0]
            im2 = ax.imshow(spatial_grid, cmap="hot", interpolation="nearest",
                            extent=extent, aspect="equal")
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
        else:
            im2 = ax.imshow(spatial_grid, cmap="hot", interpolation="nearest", aspect="equal")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
        fig.colorbar(im2, ax=ax, shrink=0.8, label="Spots/bin")
    ax.set_title("Spatial spot density")

    # --- Panel 4: pipeline funnel ---
    ax = axes[1, 1]
    stages = ["Detection", "After filter", "After dedup"]
    counts = [funnel["detection"], funnel["after_filter"], funnel["after_dedup"]]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1])
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)
    ax.set_xlabel("Spot count")
    ax.set_title("Pipeline funnel")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _q_to_prob(q):
    """Convert Phred Q-score to probability: P = 1 - 10^(-Q/10)."""
    return 1.0 - 10.0 ** (-q / 10.0)


def plot_gene_calling_qc(
    metrics: dict,
    prob: np.ndarray,
    entropy: np.ndarray,
    result_df: pd.DataFrame,
    q_loose: int = 13,
    q_standard: int = 20,
    q_strict: int = 30,
):
    """3x3 gene-calling QC composite using Q-score thresholds.

    Parameters
    ----------
    q_loose : int
        Loose Q threshold (Q13 ≈ P≥0.95). Shown in title.
    q_standard : int
        Standard Q threshold (Q20 = P≥0.99). Used for top-gene filtering.
    q_strict : int
        Strict Q threshold (Q30 = P≥0.999). Shown in title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    p_loose = _q_to_prob(q_loose)
    p_standard = _q_to_prob(q_standard)
    p_strict = _q_to_prob(q_strict)

    # Filter NaN from prob/entropy for plotting
    prob = np.where(np.isfinite(prob), prob, 0.0)
    entropy = np.where(np.isfinite(entropy), entropy, 0.0)
    total = len(prob)
    n_loose = int((prob >= p_loose).sum()) if total > 0 else 0
    n_std = int((prob >= p_standard).sum()) if total > 0 else 0
    n_strict = int((prob >= p_strict).sum()) if total > 0 else 0

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    title = (
        f'Gene Calling QC — {total:,} spots  |  '
        f'Q{q_loose}: {n_loose:,} ({n_loose / total * 100:.1f}%)  |  '
        f'Q{q_standard}: {n_std:,} ({n_std / total * 100:.1f}%)  |  '
        f'Q{q_strict}: {n_strict:,} ({n_strict / total * 100:.1f}%)'
        if total > 0
        else "Gene Calling QC — 0 spots"
    )
    fig.suptitle(title, fontsize=13)

    # (1) Probability histogram — log y, with Q-score threshold lines
    ax = axes[0, 0]
    if total > 0:
        ax.hist(prob, bins=100, color="steelblue", edgecolor="none")
        ax.axvline(p_loose, color="orange", linestyle="--", alpha=0.7, label=f"Q{q_loose} (P≥{p_loose:.2f})")
        ax.axvline(p_standard, color="red", linestyle="--", alpha=0.7, label=f"Q{q_standard} (P≥{p_standard})")
        ax.axvline(p_strict, color="darkred", linestyle="--", alpha=0.7, label=f"Q{q_strict} (P≥{p_strict})")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability distribution")
    ax.grid(True, alpha=0.3)

    # (2) Entropy histogram — log y
    ax = axes[0, 1]
    if total > 0:
        ax.hist(entropy, bins=100, color="seagreen", edgecolor="none")
        ax.set_yscale("log")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title("Entropy distribution")
    ax.grid(True, alpha=0.3)

    # (3) Probability vs Entropy 2D density — log color scale
    ax = axes[0, 2]
    if total > 0:
        h = ax.hist2d(prob, entropy, bins=100, cmap="viridis",
                      norm=LogNorm(), cmin=1)
        fig.colorbar(h[3], ax=ax, label="Count (log)")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Entropy")
    ax.set_title("Probability vs Entropy")

    # (4) Convergence (ELBO loss)
    ax = axes[1, 0]
    losses = metrics.get("convergence", {}).get("losses", [])
    if losses:
        ax.plot(range(1, len(losses) + 1), losses, color="steelblue", linewidth=1.5,
                marker="o", markersize=2, alpha=0.7)
        converged = metrics.get("convergence", {}).get("converged", False)
        status = "converged" if converged else "not converged"
        ax.set_title(f"PoSTcode ELBO ({status})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (ELBO)")
    else:
        ax.text(0.5, 0.5, "No convergence data\n(run from CSV)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("PoSTcode ELBO")
    ax.grid(True, alpha=0.3)

    # (5) Cumulative pass rate — with Q-score threshold lines
    ax = axes[1, 1]
    if total > 0:
        thresholds = np.linspace(0.0, 1.0, 101)
        pass_rate = [(prob >= t).mean() * 100 for t in thresholds]
        ax.plot(thresholds, pass_rate, color="purple", linewidth=2)
        ax.axvline(p_loose, color="orange", linestyle="--", alpha=0.7,
                   label=f"Q{q_loose}: {n_loose / total * 100:.1f}%")
        ax.axvline(p_standard, color="red", linestyle="--", alpha=0.7,
                   label=f"Q{q_standard}: {n_std / total * 100:.1f}%")
        ax.axvline(p_strict, color="darkred", linestyle="--", alpha=0.7,
                   label=f"Q{q_strict}: {n_strict / total * 100:.1f}%")
        ax.legend(fontsize=7)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Spots passing (%)")
    ax.set_title("Cumulative pass rate")
    ax.grid(True, alpha=0.3)

    # (6) Top 20 genes — filtered by Q≥q_standard (P≥0.99)
    ax = axes[1, 2]
    mask_std = prob >= p_standard
    if total > 0:
        gene_col = result_df["Gene"]
        high_genes = result_df.loc[
            mask_std & gene_col.notna()
            & ~gene_col.isin(["Background", "Infeasible"]),
            "Gene",
        ]
        if len(high_genes) > 0:
            top = high_genes.value_counts().head(20)
            ax.barh(range(len(top)), top.values[::-1], color="teal")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index[::-1], fontsize=8)
            ax.set_xlabel("Count")
            ax.set_title(f"Top 20 genes (Q≥{q_standard})")
        else:
            ax.text(0.5, 0.5, "No high-confidence gene spots",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Top genes (Q≥{q_standard})")
    else:
        ax.text(0.5, 0.5, "No spots", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis="x")

    # ----- NEW PANELS -----

    # (7) Per-gene quality scatter (Xenium signature)
    ax = axes[2, 0]
    per_gene = metrics.get("per_gene_stats", [])
    if per_gene:
        counts_arr = np.array([g["count"] for g in per_gene])
        mean_prob_arr = np.array([g["mean_prob"] for g in per_gene])
        mean_q_arr = np.array([g["mean_q"] for g in per_gene])
        sc = ax.scatter(
            np.log10(np.clip(counts_arr, 1, None)),
            mean_prob_arr,
            c=mean_q_arr,
            cmap="RdYlGn",
            s=20,
            alpha=0.8,
            vmin=0,
            vmax=40,
        )
        fig.colorbar(sc, ax=ax, label="Mean Q-score")
        ax.set_xlabel("log10(transcript count)")
        ax.set_ylabel("Mean probability")
    ax.set_title("Per-gene quality (Xenium-style)")
    ax.grid(True, alpha=0.3)

    # (8) Q-score histogram — wider x range (0–50)
    ax = axes[2, 1]
    if total > 0:
        q = compute_phred_qscore(prob)
        # Separate the cap-bin (Q==MAX) to avoid visual compression
        from sprintseq.qc.metrics import MAX_QSCORE
        q_cap = MAX_QSCORE
        q_body = q[q < q_cap]
        n_capped = int((q >= q_cap).sum())
        ax.hist(q_body, bins=np.arange(0, q_cap, 1), color="coral", edgecolor="none")
        if n_capped > 0:
            ax.bar(q_cap, n_capped, width=1, color="darkred", alpha=0.8)
            ax.annotate(f"Q≥{int(q_cap)}\n{n_capped:,}",
                        xy=(q_cap + 0.5, n_capped), fontsize=7,
                        ha="center", va="bottom")
        ax.axvline(20, color="blue", linestyle="--", alpha=0.7, label="Q=20 (1% error)")
        ax.axvline(30, color="green", linestyle="--", alpha=0.7, label="Q=30 (0.1% error)")
        ax.axvline(40, color="purple", linestyle=":", alpha=0.5, label="Q=40 (0.01% error)")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.set_xlim(0, q_cap + 3)
    ax.set_xlabel("Q-score (Phred)")
    ax.set_ylabel("Count")
    ax.set_title("Q-score distribution")
    ax.grid(True, alpha=0.3)

    # (9) Confidence margin histogram (P1 - P2) — log y
    ax = axes[2, 2]
    if total > 0 and "Probability_2" in result_df.columns:
        margin = prob - np.nan_to_num(result_df["Probability_2"].to_numpy(dtype=np.float64), nan=0.0)
        ax.hist(margin, bins=100, color="mediumpurple", edgecolor="none")
        mean_m = metrics.get("confidence_margin", {}).get("mean", np.nan)
        if np.isfinite(mean_m):
            ax.axvline(mean_m, color="red", linestyle="--", alpha=0.7,
                       label=f"Mean={mean_m:.3f}")
            ax.legend(fontsize=8)
        ax.set_yscale("log")
    ax.set_xlabel("P₁ − P₂")
    ax.set_ylabel("Count")
    ax.set_title("Confidence margin")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
