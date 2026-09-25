"""Moved to spatial-cells: `spatial-cells density`, :mod:`spatial_cells.maps.density`.

Import shim for scripts written before the split; see :mod:`sprintseq._moved`.
"""
from sprintseq._moved import forward_attr as _forward

_NEW = ("spatial_cells.maps.density", "spatial_cells.genes", "spatial_cells.quality",
        "spatial_cells.cli.density")


def __getattr__(name):
    return _forward(__name__, _NEW, name)
