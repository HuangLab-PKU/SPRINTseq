"""Tests for fill_missing_genes in density module."""

import numpy as np
import pytest
import tifffile


class TestFillMissingGenes:
    @pytest.fixture()
    def setup(self, tmp_path):
        density_dir = tmp_path / "density_0.95"
        density_dir.mkdir()
        tifffile.imwrite(str(density_dir / "CD3D.tif"), np.ones((10, 20), dtype=np.uint16))
        tifffile.imwrite(str(density_dir / "CD8A.tif"), np.full((10, 20), 5, dtype=np.uint16))

        codebook = tmp_path / "codebook.csv"
        codebook.write_text(
            "No.,Gene,Barcode\n"
            "1,CD3D,AAAACCCCGG\n"
            "2,CD8A,AAAACCCCTT\n"
            "3,GZMK,AAAAGGGGCC\n"
            "4,FOXP3,AAAAGGGGTT\n"
        )
        return str(density_dir), str(codebook)

    def test_missing_genes_filled_with_zeros(self, setup):
        from sprintseq.cli.density import fill_missing_genes
        density_dir, codebook = setup
        filled = fill_missing_genes(density_dir, codebook, (10, 20))
        assert set(filled) == {"GZMK", "FOXP3"}
        for gene in filled:
            img = tifffile.imread(f"{density_dir}/{gene}.tif")
            assert img.shape == (10, 20)
            assert img.dtype == np.uint16
            assert np.all(img == 0)

    def test_existing_genes_unchanged(self, setup):
        from sprintseq.cli.density import fill_missing_genes
        density_dir, codebook = setup
        fill_missing_genes(density_dir, codebook, (10, 20))
        assert np.all(tifffile.imread(f"{density_dir}/CD3D.tif") == 1)
        assert np.all(tifffile.imread(f"{density_dir}/CD8A.tif") == 5)

    def test_parse_gene_name_applied(self, tmp_path):
        from sprintseq.cli.density import fill_missing_genes
        density_dir = tmp_path / "d"
        density_dir.mkdir()
        tifffile.imwrite(str(density_dir / "CD3D.tif"), np.zeros((5, 5), dtype=np.uint16))
        codebook = tmp_path / "cb.csv"
        codebook.write_text("No.,Gene,Barcode\n1,SP_1_CD3D,XXX\n2,SP_2_GZMK,YYY\n")
        filled = fill_missing_genes(str(density_dir), str(codebook), (5, 5))
        assert filled == ["GZMK"]
        assert (density_dir / "GZMK.tif").exists()

    def test_no_missing_returns_empty(self, setup):
        from sprintseq.cli.density import fill_missing_genes
        density_dir, _ = setup
        codebook_full = density_dir.replace("density_0.95", "") + "cb.csv"
        with open(codebook_full, "w") as f:
            f.write("No.,Gene,Barcode\n1,CD3D,X\n2,CD8A,Y\n")
        filled = fill_missing_genes(density_dir, codebook_full, (10, 20))
        assert filled == []
