"""Spotiflow input scaling: one (mi, ma) per mosaic instead of one per block.

Spotiflow's ``normalizer="auto"`` rescales each ``predict`` call by that call's own
p1/p99.8. The readout calls it once per 2048^2 block, so the same pixel value became a
different network input depending on which block it sat in -- a tissue-free block had its
noise stretched to full range, a dense bright block pushed its dim spots down.
``SPOTIFLOW_NORMALIZATION = 'global'`` fixes (mi, ma) per mosaic.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from sprintseq.cli import readout as ro
from sprintseq.readout import spot_detection as sd


def _mosaic(h=300, w=400, pad_cols=100, seed=0):
    """Background ~200 with a few bright spots; the right `pad_cols` columns are 0 padding."""
    rng = np.random.default_rng(seed)
    img = rng.integers(150, 250, size=(h, w)).astype(np.uint16)
    for y, x in ((40, 50), (120, 200), (250, 260)):
        img[y - 1:y + 2, x - 1:x + 2] = 5000
    img[:, w - pad_cols:] = 0
    return img


class TestMosaicPercentiles:
    def test_one_window_covering_everything_is_exact(self):
        img = _mosaic()
        mi, ma = sd.mosaic_percentiles(img, window=10_000, grid=1)
        ref = np.percentile(img[img != 0], (sd.SPOTIFLOW_PMIN, sd.SPOTIFLOW_PMAX))
        assert (mi, ma) == pytest.approx(tuple(ref))

    def test_padding_is_ignored(self):
        img = _mosaic()
        mi, _ = sd.mosaic_percentiles(img, window=64, grid=6)
        assert mi >= 150  # zeros from padding would pin p1 to 0

    def test_ignore_val_none_keeps_zeros(self):
        mi, _ = sd.mosaic_percentiles(_mosaic(), window=10_000, grid=1, ignore_val=None)
        assert mi == 0

    def test_all_padding_raises(self):
        with pytest.raises(ValueError, match="ignore_val"):
            sd.mosaic_percentiles(np.zeros((64, 64), np.uint16))

    def test_windows_stay_inside_small_images(self):
        img = _mosaic(h=50, w=70, pad_cols=0)
        mi, ma = sd.mosaic_percentiles(img, window=512, grid=12)
        assert (mi, ma) == pytest.approx(tuple(np.percentile(img, (1.0, 99.8))))

    def test_reads_lazily_from_a_memmap(self, tmp_path):
        img = _mosaic()
        tifffile.imwrite(tmp_path / "m.tif", img)
        mm = tifffile.memmap(str(tmp_path / "m.tif"))
        assert sd.mosaic_percentiles(mm, window=64, grid=4) == \
            sd.mosaic_percentiles(img, window=64, grid=4)


class TestFixedRangeNormalizer:
    def test_equals_spotiflow_auto_when_given_the_image_own_percentiles(self):
        utils = pytest.importorskip("spotiflow.utils")
        img = _mosaic(pad_cols=0)
        mi, ma = np.percentile(img, (1.0, 99.8))
        np.testing.assert_array_equal(sd.fixed_range_normalizer(mi, ma)(img),
                                      utils.normalize(img))

    def test_same_value_same_output_in_every_block(self):
        pytest.importorskip("csbdeep")
        norm = sd.fixed_range_normalizer(100.0, 1100.0)
        dim = np.full((8, 8), 600, np.uint16)
        bright = dim.copy()
        bright[2:4, 2:4] = 60000
        assert norm(dim)[0, 0] == norm(bright)[0, 0] == pytest.approx(0.5)


class _RecordingModel:
    """Stands in for a Spotiflow model; records what the normalizer made of each block."""

    def __init__(self):
        self.inputs = []

    def predict(self, image, normalizer="auto", **kwargs):
        self.inputs.append(normalizer if isinstance(normalizer, str) else normalizer(image))
        return np.empty((0, 2), np.float32), None


@pytest.fixture
def recording_model(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("csbdeep")
    model = _RecordingModel()
    monkeypatch.setattr(sd, "_get_spotiflow_model", lambda *a, **k: model)
    return model


class TestGetSpotCoordinatesNormalizer:
    def test_default_keeps_spotiflow_auto(self, recording_model):
        sd.get_spot_coordinates(_mosaic(), method="spotiflow", device="cpu")
        assert recording_model.inputs == ["auto"]

    def test_norm_range_is_applied_identically_to_every_block(self, recording_model):
        img = _mosaic(pad_cols=0)
        dim_block, bright_block = img[150:250, 0:100].copy(), img[0:100, 0:100].copy()
        assert bright_block.max() == 5000 and dim_block.max() < 300
        for block in (dim_block, bright_block):
            sd.get_spot_coordinates(block, method="spotiflow", device="cpu",
                                    norm_range=(150.0, 5000.0))
        dim_in, bright_in = recording_model.inputs
        # One affine map for both blocks: identical pixel values give identical inputs.
        np.testing.assert_allclose(dim_in, (dim_block.astype(np.float32) - 150) / 4850,
                                   rtol=1e-6)
        np.testing.assert_allclose(bright_in, (bright_block.astype(np.float32) - 150) / 4850,
                                   rtol=1e-6)


class TestConfig:
    def test_default_is_a_known_mode(self):
        assert ro.SPOTIFLOW_NORMALIZATION in ro.SPOTIFLOW_NORMALIZATIONS

    def test_unknown_mode_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="spotiflow_normalization"):
            ro.detect_all_spots(tmp_path, detection_method="spotiflow",
                                spotiflow_normalization="per_tile")

    def test_cli_flag_reaches_run_pipeline(self, tmp_path, monkeypatch):
        import logging

        from sprintseq.cli.main import main

        seen = {}
        monkeypatch.setattr(ro, "BASE_DEST_DIRECTORY", str(tmp_path))
        monkeypatch.setattr(ro, "run_pipeline", lambda run_id, **kw: seen.update(kw))
        root = logging.getLogger()
        before = list(root.handlers)
        try:
            main(["readout", "--run-id", "x", "--spotiflow-normalization", "global"])
        finally:
            for h in set(root.handlers) - set(before):
                root.removeHandler(h)
                h.close()
        assert seen["spotiflow_normalization"] == "global"
        with pytest.raises(SystemExit):
            main(["readout", "--run-id", "x", "--spotiflow-normalization", "tile"])


def _hybiss_cached():
    root = Path(os.getenv("SPOTIFLOW_CACHE_DIR", Path.home() / ".spotiflow"))
    root = root if root.name == "models" else root / "models"
    return (root / ro.SPOTIFLOW_PRETRAINED_NAME).is_dir()


@pytest.mark.skipif(not _hybiss_cached(), reason="pretrained Spotiflow weights not cached")
def test_detect_all_spots_global_mode_records_the_range(tmp_path):
    """End to end through the process pool, with the real model on a tiny mosaic."""
    pytest.importorskip("spotiflow")
    stc = tmp_path / "stitched"
    stc.mkdir()
    img = _mosaic(h=256, w=256, pad_cols=64)
    spots = ((40, 50), (120, 150), (200, 100))
    for y, x in spots:
        img[y - 1:y + 2, x - 1:x + 2] = 5000
    tifffile.imwrite(stc / "cyc_1_cy3.tif", img)
    coords, stats = ro.detect_all_spots(
        stc, channels=["cy3"], detection_cycles=[1], block_size=(128, 128),
        block_overlap=(16, 16), detection_method="spotiflow", n_workers=1,
        spotiflow_normalization="global")
    assert stats["spotiflow_normalization"] == "global"
    mi, ma = stats["norm_range"]["cyc_1_cy3"]
    assert (mi, ma) == pytest.approx(sd.mosaic_percentiles(img))
    found = {(int(round(y)), int(round(x))) for y, x in coords}
    for spot in spots:
        assert any(abs(spot[0] - y) <= 1 and abs(spot[1] - x) <= 1 for y, x in found), spot
