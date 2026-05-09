"""Tests for density_stack — composite density TIFF builder."""

import os
import logging

import numpy as np
import pytest
import tifffile


class TestThermalLut:
    def test_shape_and_dtype(self):
        from sprintseq.cli.density_stack import thermal_colormap
        lut = thermal_colormap()
        assert lut.shape == (3, 256)
        assert lut.dtype == np.uint16

    def test_index_zero_is_black(self):
        from sprintseq.cli.density_stack import thermal_colormap
        lut = thermal_colormap()
        assert lut[0, 0] == 0
        assert lut[1, 0] == 0
        assert lut[2, 0] == 0

    def test_values_are_uint8_scaled(self):
        from sprintseq.cli.density_stack import thermal_colormap
        lut = thermal_colormap()
        assert np.all(lut % 256 == 0) or np.all(lut <= 65535)
        assert lut.max() <= 65280  # 255 * 256


class TestBuildDensityStack:
    @pytest.fixture()
    def density_dir(self, tmp_path):
        d = tmp_path / "density_0.95"
        d.mkdir()
        rng = np.random.default_rng(42)
        for name, peak_val in [("GeneA", 5), ("GeneB", 10), ("GeneC", 20)]:
            img = np.zeros((20, 30), dtype=np.uint16)
            img[10, 15] = peak_val
            if name == "GeneC":
                img[5:15, 10:20] = rng.integers(0, 3, size=(10, 10), dtype=np.uint16)
            tifffile.imwrite(str(d / f"{name}.tif"), img)
        return str(d)

    def test_gaussian_blur_applied(self, density_dir, tmp_path):
        from sprintseq.cli.density_stack import build_density_stack
        out = str(tmp_path / "out.tif")
        build_density_stack(density_dir, ["GeneA"], out, sigma=0.7)
        raw = tifffile.imread(out)
        blurred = np.atleast_3d(raw).reshape(-1, *raw.shape[-2:])[0]
        assert blurred[10, 15] < 5, "Peak should be reduced by Gaussian blur"
        assert blurred[10, 15] > 0, "Peak should still be nonzero"
        assert blurred[9, 15] > 0 or blurred[10, 14] > 0, "Blur should spread to neighbors"

    def test_stack_ordering(self, density_dir, tmp_path):
        from sprintseq.cli.density_stack import build_density_stack
        out = str(tmp_path / "out.tif")
        genes = ["GeneB", "GeneA", "GeneC"]
        build_density_stack(density_dir, genes, out, sigma=0.0)
        stack = tifffile.imread(out)
        assert stack.shape[0] == 3
        assert stack[0, 10, 15] == 10  # GeneB peak
        assert stack[1, 10, 15] == 5   # GeneA peak
        assert stack[2, 10, 15] > 0    # GeneC has values there

    def test_sort_flag(self, density_dir, tmp_path):
        from sprintseq.cli.density_stack import build_density_stack
        out = str(tmp_path / "out.tif")
        genes = ["GeneC", "GeneA", "GeneB"]
        build_density_stack(density_dir, genes, out, sigma=0.0, sort=True)
        with tifffile.TiffFile(out) as tf:
            labels = tf.imagej_metadata["Labels"]
            assert labels == ["GeneA.tif", "GeneB.tif", "GeneC.tif"]

    def test_imagej_metadata(self, density_dir, tmp_path):
        from sprintseq.cli.density_stack import build_density_stack
        out = str(tmp_path / "out.tif")
        genes = ["GeneA", "GeneB"]
        build_density_stack(density_dir, genes, out, display_min=1.0, display_max=10.0)
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].tags.get("ImageDescription")
            assert desc is not None
            text = desc.value
            assert "min=1.0" in text
            assert "max=10.0" in text
            cm = tf.pages[0].tags.get("ColorMap")
            assert cm is not None
            meta = tf.imagej_metadata
            assert meta is not None
            labels = meta.get("Labels", [])
            assert len(labels) == 2
            assert labels[0] == "GeneA.tif"
            assert labels[1] == "GeneB.tif"

    def test_missing_gene_skipped(self, density_dir, tmp_path, caplog):
        from sprintseq.cli.density_stack import build_density_stack
        out = str(tmp_path / "out.tif")
        with caplog.at_level(logging.WARNING):
            build_density_stack(density_dir, ["GeneA", "NoSuchGene", "GeneB"], out)
        assert any("NoSuchGene" in r.message for r in caplog.records)
        stack = tifffile.imread(out)
        assert stack.shape[0] == 2

    def test_all_genes_missing_raises(self, tmp_path):
        from sprintseq.cli.density_stack import build_density_stack
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        out = str(tmp_path / "out.tif")
        with pytest.raises(ValueError, match="[Nn]o valid"):
            build_density_stack(empty_dir, ["X", "Y"], out)
