"""Moved to spatial-cells: :mod:`spatial_cells.segment.utils`.

Import shim for scripts written before the split; see :mod:`sprintseq._moved`.
"""
from sprintseq._moved import forward_attr as _forward

_NEW = ("spatial_cells.segment.utils",)


def __getattr__(name):
    return _forward(__name__, _NEW, name)
