"""Segmentation input discovery must work on both stitched-image backends.

`auto_detect_dapi` / `auto_detect_morphology` glob for `cyc_*_<chn>.tif` and hand back
paths, which finds nothing once a run is stored as a single OME-Zarr store. The fix must
not change the signature of `prepare_cellsam_input` / `prepare_cellpose_input`: several
per-run scripts under `SPRINTseq/experiments/` call them with explicit paths, and those
have to keep working.

So the contract widens rather than changes: the prepare functions accept a path OR an
already-opened array-like handle, and the auto-detect functions return whichever suits
the backend.
"""
import numpy as np
import pytest
import tifffile

from sprintseq import segment as seg

zarr = pytest.importorskip("zarr")

SHAPE = (96, 120)


def _img(seed):
    return np.random.default_rng(seed).integers(0, 3000, size=SHAPE).astype(np.uint16)


@pytest.fixture
def tif_run(tmp_path):
    d = tmp_path / "as_tif" / "stitched"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "cyc_11_DAPI.tif", _img(1))
    tifffile.imwrite(d / "cyc_1_DAPI.tif", _img(2))
    tifffile.imwrite(d / "cyc_11_FAM.tif", _img(3))
    return d


@pytest.fixture
def zarr_run(tmp_path):
    d = tmp_path / "as_zarr" / "stitched"
    d.mkdir(parents=True)
    root = zarr.create_group(store=str(d / "mosaic.ome.zarr"), zarr_format=2, overwrite=True)
    arr = root.create_array(name="0", shape=(2, 2) + SHAPE, chunks=(1, 1, 32, 32),
                            dtype="uint16")
    arr[1, 0] = _img(1)   # cyc 11, DAPI
    arr[0, 0] = _img(2)   # cyc 1,  DAPI
    arr[1, 1] = _img(3)   # cyc 11, FAM
    root.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}, {"label": "FAM"}]}
    root.attrs["spatial_img_core"] = {"cycles": [1, 11], "channels": ["DAPI", "FAM"]}
    return d


def test_auto_detect_dapi_prefers_cycle_11_on_both_backends(tif_run, zarr_run):
    for d in (tif_run, zarr_run):
        handle = seg.auto_detect_dapi(d)
        arr = np.asarray(tifffile.memmap(str(handle))) if str(handle).endswith(".tif") \
            else np.asarray(handle)
        np.testing.assert_array_equal(arr, _img(1), err_msg=f"{d.parent.name}")


def test_auto_detect_morphology_on_both_backends(tif_run, zarr_run):
    for d in (tif_run, zarr_run):
        found = seg.auto_detect_morphology(d, names=("FAM",))
        assert len(found) == 1, f"{d.parent.name}: {found}"


def test_auto_detect_dapi_raises_when_absent(tmp_path):
    d = tmp_path / "stitched"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        seg.auto_detect_dapi(d)


def test_prepare_input_still_accepts_paths(tif_run):
    """The experiments scripts pass explicit paths; that must keep working."""
    img = seg.prepare_cellsam_input(tif_run / "cyc_11_DAPI.tif",
                                    morphology_paths=[tif_run / "cyc_11_FAM.tif"])
    assert img.shape[:2] == SHAPE


def test_prepare_input_accepts_opened_handles(zarr_run):
    dapi = seg.auto_detect_dapi(zarr_run)
    morph = seg.auto_detect_morphology(zarr_run, names=("FAM",))
    img = seg.prepare_cellsam_input(dapi, morphology_paths=morph)
    assert img.shape[:2] == SHAPE


def test_shape_mismatch_still_fails_fast(tmp_path):
    d = tmp_path / "stitched"
    d.mkdir()
    tifffile.imwrite(d / "cyc_1_DAPI.tif", _img(1))
    tifffile.imwrite(d / "cyc_1_FAM.tif", np.zeros((10, 10), np.uint16))
    with pytest.raises(ValueError, match="[Ss]hape"):
        seg.prepare_cellsam_input(d / "cyc_1_DAPI.tif",
                                  morphology_paths=[d / "cyc_1_FAM.tif"])
