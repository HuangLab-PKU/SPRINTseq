"""Moved to spatial-cells: :mod:`spatial_cells.maps.spot_geojson` (transcript points for QuPath).

Import shim for scripts written before the split; see :mod:`sprintseq._moved`.
"""
from sprintseq._moved import forward_attr as _forward

_NEW = ("spatial_cells.maps.spot_geojson",)


def __getattr__(name):
    return _forward(__name__, _NEW, name)
