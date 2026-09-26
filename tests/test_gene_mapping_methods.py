"""The decoders `map_genes` dispatches to."""
import pytest

from sprintseq.gene_calling import gene_mapping


def test_per_round_max_is_gone():
    """Removed: PerRoundMaxChannel assumes one-hot rounds, but the two-colour code has
    A = (cy3, cy5) both on and G = both off, so "brightest channel per round" cannot tell
    A from T/C or G from anything dim."""
    assert not hasattr(gene_mapping, "per_round_max_channel_mapping")
    with pytest.raises(ValueError, match="Unknown mapping method"):
        gene_mapping.map_genes(None, "unused.csv", method="per_round_max")
