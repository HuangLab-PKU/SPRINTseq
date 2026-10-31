"""QC must separate valid gene calls and unavailable probability pairs."""
import numpy as np
import pandas as pd
import pytest
from sprintseq.qc.metrics import compute_gene_calling_metrics, compute_alert_flags


def test_nan_probability_pair_does_not_destroy_finite_margin():
    frame = pd.DataFrame({"Gene": ["GeneA", "GeneA"], "Probability": [.9, np.nan], "Probability_2": [.1, np.nan], "Entropy": [.2, np.nan]})
    result = compute_gene_calling_metrics(frame)
    assert result["confidence_margin"]["mean"] == pytest.approx(.8)
    assert result["confidence_margin"]["median"] == pytest.approx(.8)
    assert result["confidence_margin"]["n_valid"] == 1
    assert result["confidence_margin"]["n_invalid"] == 1
    assert result["invalid_probability_count"] == 1
    assert result["per_gene_stats"][0]["mean_prob"] == pytest.approx(.9)
    assert result["per_gene_stats"][0]["n_valid_prob"] == 1


def test_gene_q20_excludes_confident_background_infeasible_and_decoy():
    frame = pd.DataFrame({"Gene": ["GeneA", "Background", "Infeasible", "DECOY_TEST", "GeneA"], "Probability": [.999, 1., 1., 1., np.nan], "Probability_2": [.001, 0., 0., 0., np.nan], "Entropy": [.01, 0., 0., 0., np.nan]})
    result = compute_gene_calling_metrics(frame)
    assert result["q_score"]["fraction_q20"] == pytest.approx(.8)
    assert result["q_score_gene"] == {"n_spots": 1, "n_q20": 1, "fraction_q20": 1.}


def test_unavailable_margin_is_explicit_and_alerts_do_not_crash():
    frame = pd.DataFrame({"Gene": ["GeneA"], "Probability": [np.nan], "Probability_2": [np.nan], "Entropy": [np.nan]})
    result = compute_gene_calling_metrics(frame)
    assert result["confidence_margin"]["mean"] is None
    assert result["confidence_margin"]["n_valid"] == 0
    assert result["q_score_gene"]["fraction_q20"] is None
    assert isinstance(compute_alert_flags(result), list)


def test_all_invalid_gene_probabilities_can_be_plotted():
    import matplotlib.pyplot as plt
    from sprintseq.qc.plots import plot_gene_calling_qc
    frame = pd.DataFrame({"Gene": ["GeneA"], "Probability": [np.nan], "Probability_2": [np.nan], "Entropy": [np.nan]})
    result = compute_gene_calling_metrics(frame)
    assert result["per_gene_stats"][0]["mean_prob"] is None
    figure = plot_gene_calling_qc(result, np.array([np.nan]), np.array([np.nan]), frame)
    assert figure is not None
    plt.close(figure)

def test_custom_standard_q_title_counts_only_genes_at_that_q():
    import matplotlib.pyplot as plt
    from sprintseq.qc.plots import plot_gene_calling_qc
    frame = pd.DataFrame({"Gene": ["GeneA", "GeneA", "Background"], "Probability": [.995, .9995, 1.], "Probability_2": [.004, .0004, 0.], "Entropy": [.01, .001, 0.]})
    result = compute_gene_calling_metrics(frame)
    figure = plot_gene_calling_qc(result, frame.Probability.to_numpy(), frame.Entropy.to_numpy(), frame, q_standard=30)
    assert "Q30 genes: 1 / 2 valid gene calls" in figure._suptitle.get_text()
    plt.close(figure)

def test_exact_probability_boundary_is_counted_at_standard_q():
    frame = pd.DataFrame({"Gene": ["GeneA"] * 4, "Probability": [.99, .9899, .999, .9989], "Probability_2": [.001] * 4, "Entropy": [.1] * 4})
    result = compute_gene_calling_metrics(frame)
    assert result["q_score"]["fraction_q20"] == pytest.approx(.75)
    assert result["q_score"]["fraction_q30"] == pytest.approx(.25)
    assert result["q_score_gene"]["n_q20"] == 3