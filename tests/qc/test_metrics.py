import numpy as np
import pandas as pd
import pytest

from sprintseq.qc.metrics import (
    compute_alert_flags,
    compute_density_metrics,
    compute_gene_calling_metrics,
    compute_intensity_summary,
    compute_phred_qscore,
    compute_pipeline_funnel,
    compute_spatial_density_grid,
)


# ---------- compute_phred_qscore ----------

class TestPhredQScore:
    def test_known_values(self):
        probs = np.array([0.9, 0.99, 0.999, 0.0])
        q = compute_phred_qscore(probs)
        np.testing.assert_allclose(q[0], 10.0, atol=0.01)
        np.testing.assert_allclose(q[1], 20.0, atol=0.01)
        np.testing.assert_allclose(q[2], 30.0, atol=0.01)
        np.testing.assert_allclose(q[3], 0.0, atol=0.01)

    def test_cap_at_50(self):
        q = compute_phred_qscore(np.array([1.0]))
        assert q[0] == 50.0

    def test_shape_preserved(self):
        probs = np.random.rand(100)
        q = compute_phred_qscore(probs)
        assert q.shape == probs.shape

    def test_monotonic(self):
        probs = np.linspace(0.0, 0.999, 50)
        q = compute_phred_qscore(probs)
        assert np.all(np.diff(q) >= 0)


# ---------- compute_intensity_summary ----------

class TestIntensitySummary:
    def test_keys_and_ordering(self, synthetic_intensity_df):
        result = compute_intensity_summary(synthetic_intensity_df, ["cy3", "cy5"], 4)
        for cyc in range(1, 5):
            for ch in ["cy3", "cy5"]:
                key = f"cyc_{cyc}_{ch}"
                assert key in result
                stats = result[key]
                for field in ("mean", "std", "p5", "p25", "p50", "p75", "p95"):
                    assert field in stats
                assert stats["p5"] <= stats["p25"] <= stats["p50"] <= stats["p75"] <= stats["p95"]

    def test_constant_signal(self):
        n = 100
        df = pd.DataFrame(
            {"Y": np.zeros(n), "X": np.zeros(n), "cyc_1_cy3": np.full(n, 500.0)}
        )
        result = compute_intensity_summary(df, ["cy3"], 1)
        stats = result["cyc_1_cy3"]
        assert stats["p5"] == pytest.approx(500.0)
        assert stats["p50"] == pytest.approx(500.0)
        assert stats["p95"] == pytest.approx(500.0)
        assert stats["mean"] == pytest.approx(500.0)
        assert stats["std"] == pytest.approx(0.0, abs=1e-10)


# ---------- compute_pipeline_funnel ----------

class TestPipelineFunnel:
    def test_rates(self):
        result = compute_pipeline_funnel(1000, 800, 750)
        assert result["detection"] == 1000
        assert result["after_filter"] == 800
        assert result["after_dedup"] == 750
        assert result["filter_drop_rate"] == pytest.approx(0.2)
        assert result["dedup_drop_rate"] == pytest.approx(0.0625)


# ---------- compute_spatial_density_grid ----------

class TestSpatialDensityGrid:
    def test_total_matches(self):
        rng = np.random.default_rng(0)
        n = 300
        y = rng.uniform(0, 1000, n)
        x = rng.uniform(0, 1000, n)
        grid = compute_spatial_density_grid(y, x, bin_size=200, img_shape=(1000, 1000))
        assert grid.sum() == n

    def test_shape(self):
        y = np.array([50.0, 150.0])
        x = np.array([50.0, 150.0])
        grid = compute_spatial_density_grid(y, x, bin_size=100, img_shape=(200, 300))
        assert grid.shape == (2, 3)


# ---------- compute_gene_calling_metrics ----------

class TestGeneCallingMetrics:
    def test_counts(self, synthetic_mapping_result):
        m = compute_gene_calling_metrics(synthetic_mapping_result)
        assert m["total_spots"] == 500
        assert m["background_count"] == 50
        assert m["infeasible_count"] == 50
        assert m["mapped_to_gene"] == 400

    def test_qscore(self, synthetic_mapping_result):
        m = compute_gene_calling_metrics(synthetic_mapping_result)
        assert "q_score" in m
        assert "mean" in m["q_score"]
        assert "fraction_q20" in m["q_score"]
        assert "fraction_q30" in m["q_score"]

    def test_per_gene_stats(self, synthetic_mapping_result):
        m = compute_gene_calling_metrics(synthetic_mapping_result)
        assert "per_gene_stats" in m
        assert len(m["per_gene_stats"]) > 0
        first = m["per_gene_stats"][0]
        assert "gene" in first
        assert "count" in first
        assert "mean_prob" in first
        assert "mean_q" in first

    def test_confidence_margin(self, synthetic_mapping_result):
        m = compute_gene_calling_metrics(synthetic_mapping_result)
        assert "confidence_margin" in m
        assert "mean" in m["confidence_margin"]
        assert "median" in m["confidence_margin"]

    def test_mapping_rate(self, synthetic_mapping_result):
        m = compute_gene_calling_metrics(synthetic_mapping_result)
        assert m["mapping_rate"] == pytest.approx(0.8, abs=0.01)


# ---------- compute_alert_flags ----------

class TestAlertFlags:
    def test_low_mapping_rate(self):
        metrics = {
            "mapping_rate": 0.20,
            "q_score": {"fraction_q20": 0.9},
            "confidence_margin": {"mean": 0.5},
            "background_rate": 0.05,
        }
        flags = compute_alert_flags(metrics)
        levels = [f["level"] for f in flags]
        assert "error" in levels

    def test_clean_run(self):
        metrics = {
            "mapping_rate": 0.75,
            "q_score": {"fraction_q20": 0.8},
            "confidence_margin": {"mean": 0.5},
            "background_rate": 0.05,
        }
        flags = compute_alert_flags(metrics)
        assert len(flags) == 0


# ---------- compute_density_metrics ----------

class TestDensityMetrics:
    def test_coverage_full(self):
        cube = np.ones((2, 5, 5), dtype=np.uint16)
        genes = np.array(["A", "B"])
        df = pd.DataFrame({"Y": [0], "X": [0], "Gene": ["A"], "Probability": [0.99]})
        m = compute_density_metrics(df, cube, genes, threshold=0.95, fac=200)
        assert m["spatial_coverage"] == pytest.approx(1.0)

    def test_coverage_empty(self):
        cube = np.zeros((2, 5, 5), dtype=np.uint16)
        genes = np.array(["A", "B"])
        df = pd.DataFrame({"Y": [0], "X": [0], "Gene": ["A"], "Probability": [0.99]})
        m = compute_density_metrics(df, cube, genes, threshold=0.95, fac=200)
        assert m["spatial_coverage"] == pytest.approx(0.0)

    def test_per_gene_sorted_descending(self, synthetic_density_data):
        df, cube, genes = synthetic_density_data
        m = compute_density_metrics(df, cube, genes, threshold=0.5, fac=200)
        counts = [g["count"] for g in m["per_gene_counts"]]
        assert counts == sorted(counts, reverse=True)
