"""Backend-agnostic access to stitched mosaics.

A run's stitched output is either the legacy layout -- one uncompressed TIFF per
``cyc_<N>_<chn>`` -- or a single OME-Zarr store (``mosaic.ome.zarr``, NGFF 0.4) holding
every cycle and channel in one ``(t, c, y, x)`` array. Both are opened here and both
behave the same to the caller: something you slice ``[y0:y1, x0:x1]`` out of, lazily.

**No dependency on the library that writes the store.** sprintseq has to be installable
and usable on its own, so this reads the OME-Zarr layout through the public ``zarr``
package. The FORMAT is the contract, not a shared Python module -- the same way nothing
imports a particular library just to read a TIFF. ``zarr`` is a lazy import, so a site
that only ever sees legacy TIFFs does not need it installed.
"""
from pathlib import Path

import numpy as np
import tifffile

__all__ = ["MOSAIC_STORE_NAME", "backend", "has_mosaic", "list_mosaics",
           "mosaic_shape", "open_mosaic"]

MOSAIC_STORE_NAME = "mosaic.ome.zarr"


def _store(stitch_dir):
    return Path(stitch_dir) / MOSAIC_STORE_NAME


def backend(stitch_dir):
    """``"zarr"`` if a converted store is present, else ``"tif"``.

    The store wins when both exist: conversion is additive, so the TIFFs may still be
    sitting there until someone deletes them deliberately.
    """
    return "zarr" if _store(stitch_dir).is_dir() else "tif"


def _image_meta(store, subpath):
    """``(cycles, channels, missing)`` for one image group (root, or a numbered subgroup)."""
    import json

    zattrs = (store / subpath / ".zattrs") if subpath else (store / ".zattrs")
    attrs = json.loads(zattrs.read_text(encoding="utf-8"))
    channels = [c["label"] for c in attrs.get("omero", {}).get("channels", [])]
    extra = attrs.get("spatial_img_core", {})
    cycles = extra.get("cycles")
    missing = {(int(c), ch) for c, ch in extra.get("missing", [])}
    if not cycles or not channels:
        import zarr

        arr = zarr.open(str(store), mode="r")
        shape = (arr[subpath]["0"] if subpath else arr["0"]).shape
        cycles = cycles or list(range(1, shape[0] + 1))
        channels = channels or [str(i) for i in range(shape[1])]
    return list(cycles), list(channels), missing


def _images(stitch_dir):
    """Images in a converted store, ``[(subpath, cycles, channels, missing), ...]``.

    A single-image store has one entry with ``subpath=None`` (arrays at ``root["0"]``). A
    two-image collection (root ``.zattrs`` has ``bioformats2raw.layout``: spots + morphology
    stored separately, morphology once) has one entry per numbered subgroup ``"0"``/``"1"``
    (arrays at ``root[subpath]["0"]``). Readout searches every image, so it stays
    ``(cyc, chn)``-addressable and unaware of the split.
    """
    import json

    store = _store(stitch_dir)
    attrs = json.loads((store / ".zattrs").read_text(encoding="utf-8"))
    if "bioformats2raw.layout" in attrs:
        subs = sorted((d.name for d in store.iterdir() if d.name.isdigit()), key=int)
        return [(s, *_image_meta(store, s)) for s in subs]
    return [(None, *_image_meta(store, None))]


#: Hand-made derivatives left in the stitched dir (crops, masks) that are NOT channels.
#: A channel token never ends in one of these; treating e.g. cyc_11_DAPI_crop as a channel
#: mis-reads a derived file as real signal.
_DERIVED_SUFFIXES = ("_crop", "_cut", "_mask", "_masked", "_roi", "_thumb", "_preview",
                     "_small", "_downsample", "_downsampled", "_test")


def list_mosaics(stitch_dir):
    """Every ``(cycle, channel)`` present, sorted by cycle."""
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "zarr":
        out = []
        for _, cycles, channels, missing in _images(stitch_dir):
            out += [(c, ch) for c in cycles for ch in channels if (c, ch) not in missing]
        return sorted(set(out))
    out = []
    for p in stitch_dir.glob("cyc_*.tif"):
        parts = p.stem.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit() \
                and not parts[2].lower().endswith(_DERIVED_SUFFIXES):
            out.append((int(parts[1]), parts[2]))
    return sorted(out)


def has_mosaic(stitch_dir, cyc, chn):
    """Existence check that works on either backend.

    Replaces ``(stitch_dir / f"cyc_{n}_{c}.tif").exists()``, which reports False for a
    converted run and would make the readout quietly skip cycles that are really there --
    filling their intensities with NaN instead of failing.
    """
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "zarr":
        for _, cycles, channels, missing in _images(stitch_dir):
            if cyc in cycles and chn in channels and (cyc, chn) not in missing:
                return True
        return False
    # A derived-suffix channel (DAPI_crop) has a file but is not a mosaic; stay consistent
    # with list_mosaics rather than trusting a bare file existence.
    if str(chn).lower().endswith(_DERIVED_SUFFIXES):
        return False
    return (stitch_dir / f"cyc_{cyc}_{chn}.tif").is_file()


class _ZarrPlane:
    """Lazy 2-D view of one ``(t, c)`` plane.

    Indexing the NGFF array as ``arr[t, c]`` would materialise the entire plane, which is
    several GB on a production mosaic. Reads here stay deferred until an actual slice is
    asked for, so the block loops behave as they did against ``tifffile.memmap``.
    """

    __slots__ = ("_arr", "_t", "_c")

    def __init__(self, arr, t, c):
        self._arr, self._t, self._c = arr, t, c

    @property
    def shape(self):
        return tuple(self._arr.shape[2:])

    @property
    def ndim(self):
        return self._arr.ndim - 2

    @property
    def dtype(self):
        return self._arr.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self._arr[(self._t, self._c) + key]

    def __array__(self, dtype=None, copy=None):
        out = self._arr[self._t, self._c]
        return out.astype(dtype) if dtype is not None else out


def open_mosaic(stitch_dir, cyc, chn):
    """Open one stitched mosaic for block slicing.

    Raises ``FileNotFoundError`` when the cycle/channel is absent -- a missing cycle must
    be an error the caller decides about, never a silent skip.
    """
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "zarr":
        import zarr

        root = zarr.open(str(_store(stitch_dir)), mode="r")
        for subpath, cycles, channels, missing in _images(stitch_dir):
            if cyc in cycles and chn in channels and (cyc, chn) not in missing:
                arr = root[subpath]["0"] if subpath else root["0"]
                return _ZarrPlane(arr, cycles.index(cyc), channels.index(chn))
        raise FileNotFoundError(
            f"cyc_{cyc}_{chn} not in {_store(stitch_dir)} "
            "(absent, or never acquired in any image)")

    path = stitch_dir / f"cyc_{cyc}_{chn}.tif"
    if not path.is_file():
        raise FileNotFoundError(f"cyc_{cyc}_{chn}.tif not found under {stitch_dir}")
    img = tifffile.memmap(str(path))
    return img[0] if img.ndim == 3 and img.shape[0] == 1 else img


def mosaic_shape(stitch_dir, cyc, chn):
    """Spatial ``(h, w)`` without reading pixels -- used to size the block grid."""
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "zarr":
        import zarr

        root = zarr.open(str(_store(stitch_dir)), mode="r")
        subpath = _images(stitch_dir)[0][0]     # every image shares the canvas
        arr = root[subpath]["0"] if subpath else root["0"]
        return tuple(arr.shape[-2:])
    with tifffile.TiffFile(str(stitch_dir / f"cyc_{cyc}_{chn}.tif")) as t:
        sh = tuple(t.series[0].shape)
    return sh[-2:]
