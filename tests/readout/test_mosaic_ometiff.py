"""Reading a FINALIZED run: one pyramidal ``stitched/mosaic.ome.tif``.

spatial_img_core exports a finished store to one OME-TIFF (one OME Image per store image,
levels in SubIFDs, the store's ``spatial_img_core`` attrs as JSON in each Image
Description) and deletes the store. The readout must read it exactly as it read the store
-- same ``(cycle, channel)`` addressing, same absent pairs, lazy block slicing -- without
importing spatial_img_core: the FORMAT is the contract. The fixture below writes that
format with plain tifffile; one test round-trips through the real exporter when it is
installed.
"""
import gc
import json

import numpy as np
import pytest
import tifffile

from sprintseq.readout.mosaic import (
    backend, has_mosaic, list_mosaics, mosaic_shape, open_mosaic,
)

zarr = pytest.importorskip("zarr")
pytest.importorskip("imagecodecs")

SHAPE = (300, 1100)          # wider than one 1024 tile -> ragged edge tiles


def _synthetic(cyc, chn):
    rng = np.random.default_rng((cyc * 7919 + sum(map(ord, chn))) % (2**32))
    return rng.integers(0, 4000, size=SHAPE).astype(np.uint16)


def _image(name, cycles, channels, missing):
    data = np.zeros((len(cycles), len(channels)) + SHAPE, np.uint16)
    for ti, cyc in enumerate(cycles):
        for ci, chn in enumerate(channels):
            if [cyc, chn] not in missing:
                data[ti, ci] = _synthetic(cyc, chn)
    attrs = {"multiscales": [{"name": name, "axes": [{"name": a} for a in "tcyx"]}],
             "spatial_img_core": {"cycles": cycles, "channels": channels, "missing": missing}}
    return data, attrs


def _write_finalized(d, images, description=True):
    """Write the finalized layout: images in order, level 0 + one 2x SubIFD level each."""
    opts = dict(tile=(1024, 1024), compression="zstd", predictor=True, photometric="minisblack")
    with tifffile.TiffWriter(d / "mosaic.ome.tif", bigtiff=True, ome=True) as tif:
        for data, attrs in images:
            meta = {"Name": attrs["multiscales"][0]["name"], "axes": "TCYX",
                    "Channel": {"Name": attrs["spatial_img_core"]["channels"]}}
            if description:
                meta["Description"] = json.dumps({"spatial_img_core_store": {"attrs": attrs}})
            tif.write(data, subifds=1, metadata=meta, **opts)
            tif.write(data[..., ::2, ::2], subfiletype=1, **opts)
    return d


@pytest.fixture
def ometiff_run(tmp_path):
    """Spots cy3/cy5 in cycles 1-2; morphology in cycle 3 only (T=1), FAM never acquired."""
    d = tmp_path / "finalized" / "stitched"
    d.mkdir(parents=True)
    return _write_finalized(d, [
        _image("spots", [1, 2], ["cy3", "cy5"], []),
        _image("morphology", [3], ["DAPI", "FAM"], [[3, "FAM"]]),
    ])


def test_backend_and_listing(ometiff_run):
    d = ometiff_run
    assert backend(d) == "ometiff"
    assert list_mosaics(d) == [(1, "cy3"), (1, "cy5"), (2, "cy3"), (2, "cy5"), (3, "DAPI")]
    assert has_mosaic(d, 3, "DAPI") and not has_mosaic(d, 3, "FAM") and not has_mosaic(d, 9, "cy3")
    assert mosaic_shape(d, 1, "cy3") == SHAPE


def test_reads_match_source_including_a_single_cycle_image(ometiff_run):
    d = ometiff_run
    for cyc, chn in list_mosaics(d):
        np.testing.assert_array_equal(np.asarray(open_mosaic(d, cyc, chn)), _synthetic(cyc, chn),
                                      err_msg=f"cyc {cyc} {chn}")


def test_block_slices_and_laziness(ometiff_run):
    d = ometiff_run
    a = open_mosaic(d, 2, "cy5")
    assert not isinstance(a, np.ndarray) and a.shape == SHAPE and a.ndim == 2
    ref = _synthetic(2, "cy5")
    for y, x, h, w in [(0, 0, 64, 64), (250, 1000, 128, 256), (100, 1020, 64, 64)]:
        np.testing.assert_array_equal(np.asarray(a[y:y + h, x:x + w]), ref[y:y + h, x:x + w])


def test_never_acquired_pair_fails_loudly(ometiff_run):
    with pytest.raises(FileNotFoundError, match="never acquired"):
        open_mosaic(ometiff_run, 3, "FAM")


def test_finalized_tiff_wins_over_a_leftover_store(ometiff_run):
    d = ometiff_run
    root = zarr.create_group(store=str(d / "mosaic.ome.zarr"), zarr_format=2, overwrite=True)
    root.attrs["spatial_img_core"] = {"cycles": [7], "channels": ["cy3"]}
    assert backend(d) == "ometiff" and (7, "cy3") not in list_mosaics(d)


def test_plain_ome_tiff_falls_back_to_ome_names(tmp_path):
    d = tmp_path / "plain" / "stitched"
    d.mkdir(parents=True)
    _write_finalized(d, [_image("spots", [1, 2], ["cy3", "cy5"], [])], description=False)
    assert list_mosaics(d) == [(1, "cy3"), (1, "cy5"), (2, "cy3"), (2, "cy5")]


def test_dropped_handle_releases_the_file(ometiff_run):
    """Windows cannot delete an open file; a dropped handle must not pin the TIFF."""
    d = ometiff_run
    np.asarray(open_mosaic(d, 1, "cy3")[0:4, 0:4])
    gc.collect()
    (d / "mosaic.ome.tif").unlink()


def test_segmentation_autodetect_uses_the_finalized_tiff(ometiff_run):
    from sprintseq import segment as seg

    dapi = seg.auto_detect_dapi(ometiff_run)
    assert not isinstance(dapi, (str, np.ndarray)) and dapi.shape == SHAPE
    assert seg.auto_detect_morphology(ometiff_run, names=("FAM",)) == []


def test_contract_with_the_real_exporter(tmp_path):
    """End to end: spatial_img_core writes a store, finalizes it, sprintseq reads it."""
    mio = pytest.importorskip("spatial_img_core.mosaic_io")
    mo = pytest.importorskip("spatial_img_core.mosaic_ometiff")
    d = tmp_path / "20990101_TEST_processed" / "stitched"
    d.mkdir(parents=True)
    roles = {"spot": ["cy3", "cy5"], "morphology": ["DAPI"], "spot_cycles": None, "overrides": {}}
    pairs = [(1, "cy3"), (1, "cy5"), (2, "cy3"), (2, "cy5"), (3, "DAPI")]
    w = mio.MosaicWriter(d, pairs, SHAPE, np.uint16, roles=roles)
    for cyc, chn in pairs:
        w.write_plane(cyc, chn, _synthetic(cyc, chn))
    w.finalize()
    mo.finalize_mosaic(d)
    assert backend(d) == "ometiff" and not (d / "mosaic.ome.zarr").exists()
    assert list_mosaics(d) == sorted(pairs)
    for cyc, chn in pairs:
        np.testing.assert_array_equal(np.asarray(open_mosaic(d, cyc, chn)), _synthetic(cyc, chn))
