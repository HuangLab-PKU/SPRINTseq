"""Backend-agnostic access to stitched mosaics.

A run's stitched output is one of three layouts, all opened here and all behaving the same
to the caller -- something you slice ``[y0:y1, x0:x1]`` out of, lazily:

- **ometiff** -- a FINALIZED run: one pyramidal ``mosaic.ome.tif`` (tiled, zstd), one OME
  Image per store image (spots, morphology), exported from the store once the run is
  final. ~52k chunk files become one, reads over SMB are 1.4-3x faster, and QuPath opens it
  in seconds. Wins over everything else when present.
- **zarr** -- the working store ``mosaic.ome.zarr`` (NGFF 0.4) holding every cycle and
  channel in ``(t, c, y, x)`` arrays, while a run is still being stitched or edited.
- **tif** -- the legacy layout, one uncompressed TIFF per ``cyc_<N>_<chn>``.

**No dependency on the library that writes them** (spatial_img_core). sprintseq has to be
installable and usable on its own, so the FORMAT is the contract: the store is read through
the public ``zarr`` package, the finalized TIFF through ``tifffile``. Cycle numbers and
never-acquired pairs of a finalized run come from the JSON its writer puts in each OME
Image ``Description`` (key ``spatial_img_core_store``), which carries the store's
``spatial_img_core`` attrs verbatim. ``zarr`` is a lazy import, so a site that only ever
sees legacy TIFFs does not need it installed.
"""
import functools
import json
import weakref
from pathlib import Path

import numpy as np
import tifffile

__all__ = ["MOSAIC_STORE_NAME", "MOSAIC_TIFF_NAME", "backend", "has_mosaic", "list_mosaics",
           "mosaic_shape", "open_mosaic"]

MOSAIC_STORE_NAME = "mosaic.ome.zarr"
MOSAIC_TIFF_NAME = "mosaic.ome.tif"
_DESCRIPTION_KEY = "spatial_img_core_store"


def _store(stitch_dir):
    return Path(stitch_dir) / MOSAIC_STORE_NAME


def _tiff(stitch_dir):
    return Path(stitch_dir) / MOSAIC_TIFF_NAME


def backend(stitch_dir):
    """``"ometiff"`` for a finalized run, ``"zarr"`` if a store is present, else ``"tif"``.

    The finalized TIFF wins over a store (it was verified against the store before the
    store was deleted, and nothing edits a store beside it); the store wins over legacy
    per-cycle TIFFs, which may still be sitting there after conversion.
    """
    if _tiff(stitch_dir).is_file():
        return "ometiff"
    return "zarr" if _store(stitch_dir).is_dir() else "tif"


@functools.lru_cache(maxsize=64)
def _tiff_images_cached(path, mtime_ns, size):
    """``((cycles, channels, missing, is_3d), ...)`` per OME Image, in series order."""
    import xml.etree.ElementTree as ET

    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata
        shapes = [s.get_shape(False) for s in tf.series]
    if not xml:
        raise ValueError(f"{path} carries no OME-XML")
    ome = ET.fromstring(xml)
    ns = {"o": ome.tag.split("}")[0].strip("{")}
    out = []
    for i, img in enumerate(ome.findall("o:Image", ns)):
        channels = [c.get("Name") for c in img.iter(f"{{{ns['o']}}}Channel")]
        desc = img.find("o:Description", ns)
        extra, is_3d = {}, shapes[i][2] > 1
        try:
            attrs = json.loads(desc.text)[_DESCRIPTION_KEY]["attrs"]
            extra = attrs.get("spatial_img_core", {})
            is_3d = any(a["name"] == "z" for a in attrs["multiscales"][0]["axes"])
        except (AttributeError, TypeError, ValueError, KeyError):
            pass                       # a plain OME-TIFF: fall back to OME + shape
        channels = list(extra.get("channels") or channels)
        cycles = list(extra.get("cycles") or range(1, shapes[i][0] + 1))
        missing = frozenset((int(c), ch) for c, ch in extra.get("missing", []))
        out.append((cycles, channels, missing, is_3d))
    return tuple(out)


def _tiff_images(stitch_dir):
    p = _tiff(stitch_dir)
    st = p.stat()
    return _tiff_images_cached(str(p), st.st_mtime_ns, st.st_size)


class _OmeTiffLevel0:
    """Level 0 of one image in ``mosaic.ome.tif``, indexed like the store's ``(t, c, y, x)``
    array. tifffile exposes the OME image unsqueezed as 6-D ``(T, C, Z, Y, X, S)`` -- read
    squeezed, a length-1 T or C would vanish and shift every index. Closes the file when
    garbage-collected (tifffile objects form reference cycles; on Windows an open handle
    blocks deleting or replacing the file)."""

    __slots__ = ("_arr", "_is_3d", "_keep", "__weakref__")

    def __init__(self, path, image, is_3d):
        import zarr

        tf = tifffile.TiffFile(path)
        zstore = tf.aszarr(series=image, level=0, squeeze=False)
        self._arr, self._is_3d, self._keep = zarr.open(zstore, mode="r"), is_3d, (tf, zstore)
        weakref.finalize(self, tf.close)

    @property
    def shape(self):
        t, c, z, y, x, _ = self._arr.shape
        return (t, c, z, y, x) if self._is_3d else (t, c, y, x)

    @property
    def ndim(self):
        return 5 if self._is_3d else 4

    @property
    def dtype(self):
        return self._arr.dtype

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        if not self._is_3d:
            key = key[:2] + (0,) + key[2:]
        return self._arr[key + (0,)]


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


#: Hand-made derivatives left in the stitched dir (crops, masks, filtered copies) that are
#: NOT channels. A channel token never ends in one of these; treating e.g. cyc_11_DAPI_crop
#: or cyc_1_cy3_tophat as a channel mis-reads a derived file as real signal. Keep in sync
#: with spatial_img_core.mosaic_io._DERIVED_SUFFIXES.
_DERIVED_SUFFIXES = ("_crop", "_cut", "_mask", "_masked", "_roi", "_thumb", "_preview",
                     "_small", "_downsample", "_downsampled", "_test", "_tophat",
                     "_filtered", "_bgsub", "_norm")


def list_mosaics(stitch_dir):
    """Every ``(cycle, channel)`` present, sorted by cycle."""
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "ometiff":
        out = []
        for cycles, channels, missing, _ in _tiff_images(stitch_dir):
            out += [(c, ch) for c in cycles for ch in channels if (c, ch) not in missing]
        return sorted(set(out))
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
    if backend(stitch_dir) == "ometiff":
        return any(cyc in cycles and chn in channels and (cyc, chn) not in missing
                   for cycles, channels, missing, _ in _tiff_images(stitch_dir))
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
    if backend(stitch_dir) == "ometiff":
        for i, (cycles, channels, missing, is_3d) in enumerate(_tiff_images(stitch_dir)):
            if cyc in cycles and chn in channels and (cyc, chn) not in missing:
                return _ZarrPlane(_OmeTiffLevel0(str(_tiff(stitch_dir)), i, is_3d),
                                  cycles.index(cyc), channels.index(chn))
        raise FileNotFoundError(
            f"cyc_{cyc}_{chn} not in {_tiff(stitch_dir)} "
            "(absent, or never acquired in any image)")
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
    if backend(stitch_dir) == "ometiff":
        with tifffile.TiffFile(str(_tiff(stitch_dir))) as t:   # every image shares the canvas
            return tuple(t.series[0].get_shape(False)[3:5])
    if backend(stitch_dir) == "zarr":
        import zarr

        root = zarr.open(str(_store(stitch_dir)), mode="r")
        subpath = _images(stitch_dir)[0][0]     # every image shares the canvas
        arr = root[subpath]["0"] if subpath else root["0"]
        return tuple(arr.shape[-2:])
    with tifffile.TiffFile(str(stitch_dir / f"cyc_{cyc}_{chn}.tif")) as t:
        sh = tuple(t.series[0].shape)
    return sh[-2:]
