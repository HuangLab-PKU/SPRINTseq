"""Moved to spatial-cells: `spatial-cells density-stack`, :mod:`spatial_cells.maps.density_stack`.

Import shim for scripts written before the split; see :mod:`sprintseq._moved`.
"""
from sprintseq._moved import forward_attr as _forward

_NEW = ("spatial_cells.maps.density_stack", "spatial_cells.quality",
        "spatial_cells.cli.density_stack")


def __getattr__(name):
    return _forward(__name__, _NEW, name)
