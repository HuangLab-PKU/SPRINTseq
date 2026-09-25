from .report import generate_readout_qc, generate_gene_calling_qc

__all__ = [
    "generate_readout_qc",
    "generate_gene_calling_qc",
]


def __getattr__(name):
    # Density QC moved to spatial-cells with the density stage (see sprintseq._moved).
    from sprintseq._moved import forward_attr
    return forward_attr(__name__, ("spatial_cells.qc",), name)
