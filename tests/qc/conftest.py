import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_intensity_df():
    """500 spots, 2 channels (cy3, cy5), 4 cycles. Lognormal intensities."""
    rng = np.random.default_rng(42)
    n = 500
    data = {"Y": rng.uniform(0, 10000, n), "X": rng.uniform(0, 10000, n)}
    for cyc in range(1, 5):
        for ch in ["cy3", "cy5"]:
            data[f"cyc_{cyc}_{ch}"] = rng.lognormal(mean=6, sigma=1, size=n).astype(
                np.float64
            )
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_position_df():
    """500 spots with index, Y, X."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {"index": np.arange(n), "Y": rng.uniform(0, 10000, n), "X": rng.uniform(0, 10000, n)}
    )


@pytest.fixture
def synthetic_mapping_result():
    """500 spots: 80% gene, 10% Background, 10% Infeasible."""
    rng = np.random.default_rng(42)
    n = 500
    genes = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]
    gene_col = np.array(
        [genes[i % len(genes)] for i in range(int(n * 0.8))]
        + ["Background"] * int(n * 0.1)
        + ["Infeasible"] * int(n * 0.1)
    )
    rng.shuffle(gene_col)
    prob = rng.beta(5, 2, size=n).astype(np.float64)
    prob_2 = prob * rng.uniform(0.1, 0.4, size=n)
    entropy = -prob * np.log(np.clip(prob, 1e-10, 1))
    return pd.DataFrame(
        {
            "index": np.arange(n),
            "Sequence": ["ACGT"] * n,
            "Gene": gene_col,
            "Probability": prob,
            "Probability_2": prob_2,
            "Entropy": entropy,
        }
    )


@pytest.fixture
def synthetic_density_data():
    """200 filtered spots, 3 genes, 10x10 density cube."""
    rng = np.random.default_rng(42)
    n = 200
    genes = np.array(["GeneA", "GeneB", "GeneC"])
    gene_col = genes[rng.integers(0, 3, size=n)]
    df = pd.DataFrame(
        {
            "Y": rng.uniform(0, 2000, n),
            "X": rng.uniform(0, 2000, n),
            "Gene": gene_col,
            "Probability": rng.beta(8, 2, size=n),
        }
    )
    density_cube = rng.integers(0, 10, size=(3, 10, 10), dtype=np.uint16)
    return df, density_cube, genes
