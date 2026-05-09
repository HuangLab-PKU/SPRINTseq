import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sprintseq.qc.plots import (
    plot_density_qc,
    plot_gene_calling_qc,
    plot_readout_qc,
)


@pytest.fixture
def readout_qc_inputs():
    intensity_summary = {}
    for cyc in range(1, 5):
        for ch in ["cy3", "cy5"]:
            intensity_summary[f"cyc_{cyc}_{ch}"] = {
                "mean": 500.0, "std": 100.0,
                "p5": 300.0, "p25": 400.0, "p50": 500.0, "p75": 600.0, "p95": 700.0,
            }
    funnel = {
        "detection": 10000, "after_filter": 8000, "after_dedup": 7500,
        "filter_drop_rate": 0.2, "dedup_drop_rate": 0.0625,
    }
    spatial_grid = np.random.default_rng(0).integers(0, 50, (10, 10))
    return intensity_summary, funnel, spatial_grid


@pytest.fixture
def gene_calling_qc_inputs(synthetic_mapping_result):
    from sprintseq.qc.metrics import compute_gene_calling_metrics
    metrics = compute_gene_calling_metrics(synthetic_mapping_result)
    metrics["convergence"] = {"losses": [], "converged": False, "n_iterations": 0, "final_loss": None}
    prob = synthetic_mapping_result["Probability"].to_numpy()
    entropy = synthetic_mapping_result["Entropy"].to_numpy()
    return metrics, prob, entropy, synthetic_mapping_result


@pytest.fixture
def density_qc_inputs():
    per_gene_counts = [
        {"gene": "GeneA", "count": 100},
        {"gene": "GeneB", "count": 80},
        {"gene": "GeneC", "count": 50},
    ]
    total_density = np.random.default_rng(0).integers(0, 20, (10, 10))
    return per_gene_counts, total_density


class TestPlotReadoutQC:
    def test_returns_figure(self, readout_qc_inputs):
        summary, funnel, grid = readout_qc_inputs
        fig = plot_readout_qc(summary, funnel, grid, ["cy3", "cy5"], 4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_count(self, readout_qc_inputs):
        summary, funnel, grid = readout_qc_inputs
        fig = plot_readout_qc(summary, funnel, grid, ["cy3", "cy5"], 4)
        assert len(fig.axes) >= 4
        plt.close(fig)


class TestPlotGeneCallingQC:
    def test_returns_figure(self, gene_calling_qc_inputs):
        metrics, prob, entropy, df = gene_calling_qc_inputs
        fig = plot_gene_calling_qc(metrics, prob, entropy, df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_count(self, gene_calling_qc_inputs):
        metrics, prob, entropy, df = gene_calling_qc_inputs
        fig = plot_gene_calling_qc(metrics, prob, entropy, df)
        assert len(fig.axes) >= 9
        plt.close(fig)

    def test_handles_all_background(self):
        from sprintseq.qc.metrics import compute_gene_calling_metrics
        n = 100
        df = pd.DataFrame({
            "index": range(n), "Sequence": ["ACGT"] * n,
            "Gene": ["Background"] * n,
            "Probability": np.full(n, 0.5),
            "Probability_2": np.full(n, 0.1),
            "Entropy": np.full(n, 0.5),
        })
        metrics = compute_gene_calling_metrics(df)
        metrics["convergence"] = {"losses": [], "converged": False, "n_iterations": 0, "final_loss": None}
        fig = plot_gene_calling_qc(
            metrics, df["Probability"].values, df["Entropy"].values, df
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotDensityQC:
    def test_returns_figure(self, density_qc_inputs):
        counts, density = density_qc_inputs
        fig = plot_density_qc(counts, density)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_count(self, density_qc_inputs):
        counts, density = density_qc_inputs
        fig = plot_density_qc(counts, density)
        assert len(fig.axes) >= 2
        plt.close(fig)
