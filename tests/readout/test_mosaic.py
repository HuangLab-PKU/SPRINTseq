"""Backend-agnostic mosaic access, with no dependency on spatial_img_core.

The readout pipeline must keep working when sprintseq is installed on its own, so it
reads the OME-Zarr layout through the public `zarr` package rather than importing the
library that writes it. The FORMAT is the contract between the two repos, not a shared
Python module -- the same way anything else reads a TIFF without importing whatever
produced it.

`zarr` itself stays optional: a site that only ever sees legacy per-channel TIFFs should
not have to install it.
"""
import json

import numpy as np
import pytest
import tifffile

from sprintseq.readout.mosaic import (
    backend, has_mosaic, list_mosaics, mosaic_shape, open_mosaic,
)

zarr = pytest.importorskip("zarr")

CHANNELS = ["cy3", "cy5"]
CYCLES = [1, 2]
SHAPE = (140, 260)


def _synthetic(cyc, chn):
    rng = np.random.default_rng(abs(hash((cyc, chn))) % (2**32))
    return rng.integers(0, 4000, size=SHAPE).astype(np.uint16)


@pytest.fixture
def tif_run(tmp_path):
    # Distinct subdirectory per fixture: several tests take BOTH, and a shared
    # tmp_path/"stitched" would collide on the second mkdir.
    d = tmp_path / "as_tif" / "stitched"
    d.mkdir(parents=True)
    for cyc in CYCLES:
        for chn in CHANNELS:
            tifffile.imwrite(d / f"cyc_{cyc}_{chn}.tif", _synthetic(cyc, chn))
    return d


@pytest.fixture
def zarr_run(tmp_path):
    """A store written the way spatial_img_core writes one -- NGFF 0.4, zarr v2 layout."""
    d = tmp_path / "as_zarr" / "stitched"
    d.mkdir(parents=True)
    store = d / "mosaic.ome.zarr"
    root = zarr.create_group(store=str(store), zarr_format=2, overwrite=True)
    arr = root.create_array(name="0", shape=(len(CYCLES), len(CHANNELS)) + SHAPE,
                            chunks=(1, 1, 64, 64), dtype="uint16")
    for ti, cyc in enumerate(CYCLES):
        for ci, chn in enumerate(CHANNELS):
            arr[ti, ci] = _synthetic(cyc, chn)
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": "t"}, {"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "0"}],
    }]
    root.attrs["omero"] = {"channels": [{"label": c} for c in CHANNELS]}
    root.attrs["spatial_img_core"] = {"cycles": CYCLES, "channels": CHANNELS}
    return d


def test_backend_detection(tif_run, zarr_run):
    assert backend(tif_run) == "tif"
    assert backend(zarr_run) == "zarr"


def test_reads_two_image_collection(tmp_path):
    """readout must read a spots+morphology collection transparently: (cyc,chn) routes to
    the right subgroup, morphology stored once (not per-t)."""
    d = tmp_path / "stitched"
    d.mkdir()
    store = d / "mosaic.ome.zarr"
    root = zarr.create_group(store=str(store), zarr_format=2, overwrite=True)
    # series 0 = spots (t=2, c=[cy3,cy5]); series 1 = morphology (t=1, c=[DAPI])
    s0 = root.create_group(name="0")
    a0 = s0.create_array(name="0", shape=(2, 2) + SHAPE, chunks=(1, 1, 64, 64), dtype="uint16")
    a0[0, 0] = _synthetic(1, "cy3"); a0[0, 1] = _synthetic(1, "cy5")
    a0[1, 0] = _synthetic(2, "cy3"); a0[1, 1] = _synthetic(2, "cy5")
    s0.attrs["multiscales"] = [{"version": "0.4", "name": "spots", "datasets": [{"path": "0"}]}]
    s0.attrs["omero"] = {"channels": [{"label": "cy3"}, {"label": "cy5"}]}
    s0.attrs["spatial_img_core"] = {"cycles": [1, 2], "channels": ["cy3", "cy5"], "missing": []}
    s1 = root.create_group(name="1")
    a1 = s1.create_array(name="0", shape=(1, 1) + SHAPE, chunks=(1, 1, 64, 64), dtype="uint16")
    a1[0, 0] = _synthetic(2, "DAPI")
    s1.attrs["multiscales"] = [{"version": "0.4", "name": "morphology", "datasets": [{"path": "0"}]}]
    s1.attrs["omero"] = {"channels": [{"label": "DAPI"}]}
    s1.attrs["spatial_img_core"] = {"cycles": [2], "channels": ["DAPI"], "missing": []}
    root.attrs["bioformats2raw.layout"] = 3

    assert set(list_mosaics(d)) == {(1, "cy3"), (1, "cy5"), (2, "cy3"), (2, "cy5"), (2, "DAPI")}
    assert has_mosaic(d, 2, "DAPI") and not has_mosaic(d, 1, "DAPI")
    np.testing.assert_array_equal(np.asarray(open_mosaic(d, 1, "cy3")), _synthetic(1, "cy3"))
    np.testing.assert_array_equal(np.asarray(open_mosaic(d, 2, "DAPI")), _synthetic(2, "DAPI"))
    assert mosaic_shape(d, 1, "cy3") == SHAPE


def test_derived_tifs_are_not_mosaics(tmp_path):
    """cyc_11_DAPI_crop.tif and friends are hand-made derivatives, not channels; picking
    one up as a channel feeds a derived file into readout as if it were real signal."""
    d = tmp_path / "stitched"
    d.mkdir()
    tifffile.imwrite(d / "cyc_1_cy3.tif", _synthetic(1, "cy3"))
    tifffile.imwrite(d / "cyc_1_cy5.tif", _synthetic(1, "cy5"))
    tifffile.imwrite(d / "cyc_11_DAPI_crop.tif", _synthetic(11, "DAPI")[:32, :32])
    tifffile.imwrite(d / "cyc_1_cy3_masked.tif", _synthetic(1, "cy3")[:32, :32])
    assert set(list_mosaics(d)) == {(1, "cy3"), (1, "cy5")}
    assert not has_mosaic(d, 11, "DAPI_crop")


def test_reads_match_source_on_both_backends(tif_run, zarr_run):
    for d in (tif_run, zarr_run):
        for cyc in CYCLES:
            for chn in CHANNELS:
                np.testing.assert_array_equal(
                    np.asarray(open_mosaic(d, cyc, chn)), _synthetic(cyc, chn),
                    err_msg=f"{backend(d)} backend, cyc {cyc} {chn}")


def test_block_slices_agree_across_backends(tif_run, zarr_run):
    """The readout loop only ever does a[y0:y1, x0:x1]; both backends must match there,
    including at ragged edges past the array bound."""
    blocks = [(0, 0, 64, 64), (60, 60, 128, 128), (100, 200, 64, 128)]
    for cyc in CYCLES:
        for chn in CHANNELS:
            a = open_mosaic(tif_run, cyc, chn)
            b = open_mosaic(zarr_run, cyc, chn)
            for y, x, h, w in blocks:
                np.testing.assert_array_equal(
                    np.asarray(a[y:y + h, x:x + w]), np.asarray(b[y:y + h, x:x + w]),
                    err_msg=f"block ({y},{x},{h},{w}) cyc {cyc} {chn}")


def test_zarr_reads_stay_lazy(zarr_run):
    """Indexing the handle must not materialise the whole plane -- production mosaics are
    multi-GB and the readout walks them a block at a time."""
    a = open_mosaic(zarr_run, 1, "cy3")
    assert not isinstance(a, np.ndarray), "handle should be a lazy view, not an array"
    assert a.shape == SHAPE
    assert np.asarray(a[0:8, 0:8]).shape == (8, 8)


def test_shape_without_reading_pixels(tif_run, zarr_run):
    assert mosaic_shape(tif_run, 1, "cy3") == SHAPE
    assert mosaic_shape(zarr_run, 1, "cy3") == SHAPE


def test_has_mosaic(tif_run, zarr_run):
    for d in (tif_run, zarr_run):
        assert has_mosaic(d, 1, "cy3")
        assert not has_mosaic(d, 9, "cy3")
        assert not has_mosaic(d, 1, "nope")


def test_missing_fails_loudly(tif_run, zarr_run):
    for d in (tif_run, zarr_run):
        with pytest.raises(FileNotFoundError):
            open_mosaic(d, 9, "cy3")


def test_zero_filled_planes_are_not_reported_as_present(tmp_path):
    """A sparse panel must not gain phantom mosaics.

    The store is a dense (t, c, y, x) array, so a 12-cycle x 4-channel SPRINTseq run with
    only 25 acquired planes allocates all 48 and zero-fills 23 of them. If discovery
    returns the full cross product the readout reads zeros for an absent cycle and records
    them as real intensities -- silent corruption, not a visible failure.
    """
    d = tmp_path / "stitched"
    d.mkdir()
    store = d / "mosaic.ome.zarr"
    root = zarr.create_group(store=str(store), zarr_format=2, overwrite=True)
    root.create_array(name="0", shape=(2, 2) + SHAPE, chunks=(1, 1, 64, 64), dtype="uint16")
    root.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
    root.attrs["omero"] = {"channels": [{"label": "cy3"}, {"label": "DAPI"}]}
    root.attrs["spatial_img_core"] = {
        "cycles": [1, 2], "channels": ["cy3", "DAPI"],
        "missing": [[1, "DAPI"]],          # DAPI only acquired on cycle 2
    }

    assert set(list_mosaics(d)) == {(1, "cy3"), (2, "cy3"), (2, "DAPI")}
    assert not has_mosaic(d, 1, "DAPI")
    assert has_mosaic(d, 2, "DAPI")
    with pytest.raises(FileNotFoundError, match="never acquired"):
        open_mosaic(d, 1, "DAPI")


def test_channels_recovered_from_standard_omero_metadata(tmp_path):
    """Must not depend on the writer's private attrs block: fall back to the NGFF
    `omero.channels[].label` list, which any OME-Zarr writer produces."""
    d = tmp_path / "stitched"
    d.mkdir()
    store = d / "mosaic.ome.zarr"
    root = zarr.create_group(store=str(store), zarr_format=2, overwrite=True)
    root.create_array(name="0", shape=(1, 2) + SHAPE, chunks=(1, 1, 64, 64), dtype="uint16")
    root.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
    root.attrs["omero"] = {"channels": [{"label": c} for c in CHANNELS]}
    # no "spatial_img_core" block at all
    assert has_mosaic(d, 1, "cy3")
    assert open_mosaic(d, 1, "cy5").shape == SHAPE
