"""Per-cell gene maps (`sprintseq cell-map`): counts painted onto real cell footprints.

The contract that matters downstream: the viewer's pixel value is the per-cell count and
every channel / slice is named by gene. For the pyramidal OME-TIFF (read tile by tile by
QuPath and Bio-Formats) that means OME channel names, a physical pixel size, and levels
that are each painted from the mask -- integer counts, every cell whole -- rather than
averaged. For ImageJ-ZIP it means slice labels, LUT, display range and um calibration
surviving the zip container, since a compressed TIFF stack loses its slice labels in
ImageJ. A mask that does not belong to the spot table must fail loudly, not paint counts
onto the wrong cells.
"""
import argparse
import io
import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import pytest
import tifffile

from sprintseq.cli import cellmap as cli
from sprintseq.segment import cellmap as cm

pytest.importorskip("zarr")

RUN = "20990101_TEST_cellmap"
FAC = 10


def _mask():
    m = np.zeros((200, 240), dtype=np.uint32)
    m[20:60, 20:60] = 1      # big cell
    m[20:60, 60:100] = 2     # touches cell 1 along x = 60
    m[150:153, 150:153] = 3  # smaller than a 10 x 10 block: no sample point lands on it
    m[100:140, 150:200] = 4  # a cell without spots
    return m


def _spots():
    rows = (
        [(40.2, 40.7, 'GeneA', 1.0, 1)] * 3
        + [(30.0, 30.0, 'GeneB', 0.995, 1),
           (31.0, 31.0, 'GeneB', 0.5, 1),          # below Q20
           (32.0, 32.0, 'Background', 1.0, 1),     # not a gene
           (33.0, 33.0, 'Infeasible', 1.0, 1),
           (40.0, 80.0, 'GeneA', 0.999, 2),
           (41.0, 81.0, 'SP_12_GeneC', 1.0, 2),    # prefix stripped like density
           (42.0, 82.0, 'SP_12_GeneC', 1.0, 2)]
        + [(151.5, 151.5, 'GeneB', 1.0, 3)] * 4
    )
    df = pd.DataFrame(rows, columns=['Y', 'X', 'Gene', 'Probability', 'Cell_ID'])
    df['fov_masked'] = False
    df.loc[df.index[-5], 'fov_masked'] = True     # one of cell 2's GeneC spots
    return df


@pytest.fixture
def run_dir(tmp_path):
    seg = tmp_path / f"{RUN}_processed" / "segmented"
    seg.mkdir(parents=True)
    tifffile.imwrite(seg / "cellsam_mask.tif", _mask(), tile=(64, 64), compression='zlib')
    _spots().to_csv(seg / "assigned_spots.csv", index=False)
    (tmp_path / "genes.txt").write_text("GeneA\nGeneC\nGeneB\nGeneZ\n", encoding="utf-8")
    return tmp_path


def _read_zip(path):
    with zipfile.ZipFile(path) as zf:
        (name,) = zf.namelist()
        assert name == path.stem + ".tif"
        with tifffile.TiffFile(io.BytesIO(zf.read(name))) as tf:
            page = tf.pages[0]
            return (tf.asarray(), tf.imagej_metadata, page.compression,
                    page.tags['XResolution'].value, page.colormap)


def _read_ome(path):
    """Every pyramid level as an array, plus the OME channel names and pixel size."""
    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome and tf.is_bigtiff
        series = tf.series[0]
        assert series.axes == 'CYX'
        levels = [lv.asarray() for lv in series.levels]
        ome = ET.fromstring(tf.ome_metadata)
        ns = {'o': ome.tag.split('}')[0].strip('{')}
        pixels = ome.find('.//o:Pixels', ns)
        names = [c.get('Name') for c in pixels.findall('o:Channel', ns)]
        return levels, names, float(pixels.get('PhysicalSizeX')), series.levels[0].keyframe


def _run(run_dir, **kw):
    kw.setdefault('gene_file', str(run_dir / "genes.txt"))
    kw.setdefault('fac', FAC)
    return cli.run_pipeline(RUN, threshold=0.99, label="Q20", base_dir=str(run_dir), **kw)


class TestSpotTable:
    def test_filter_matches_density_cut(self):
        out = cm.filter_spots(_spots(), 0.99)
        assert set(out['Gene']) == {'GeneA', 'GeneB', 'GeneC'}   # prefix stripped, non-genes gone
        assert (out['Probability'] > 0.99).all()
        assert len(out) == 3 + 1 + 1 + 2 + 4

    def test_fov_masked_spots_dropped_on_request(self):
        assert len(cm.filter_spots(_spots(), 0.99, exclude_fov_masked=True)) == 10

    def test_fov_masked_requires_the_column(self):
        with pytest.raises(ValueError, match="fov_masked"):
            cm.filter_spots(_spots().drop(columns='fov_masked'), 0.99, exclude_fov_masked=True)

    def test_counts_and_totals(self):
        spots = cm.filter_spots(_spots(), 0.99)
        ids, counts, totals, cy, cx = cm.cell_gene_counts(spots, ['GeneA', 'GeneB', 'GeneC', 'GeneZ'])
        assert ids.tolist() == [1, 2, 3]
        assert counts.tolist() == [[3, 1, 0, 0], [1, 0, 2, 0], [0, 4, 0, 0]]
        assert totals.tolist() == [4, 3, 4]
        assert cy[2] == pytest.approx(151.5) and cx[2] == pytest.approx(151.5)


class TestLabelMap:
    def test_border_separates_touching_cells_only_where_they_touch(self):
        lab = cm.downsample_labels(_mask(), FAC, (0, 200, 0, 240))
        sep = cm.separate_touching_cells(lab, min_area=9)
        assert lab[4, 6] == 2 and sep[4, 6] == 0     # first column of cell 2, next to cell 1
        assert sep[4, 7] == 2 and sep[4, 5] == 1     # both cells keep their interiors
        assert (sep[lab == 4] == 4).all()            # a free-standing cell is untouched

    def test_small_cells_keep_every_pixel(self):
        lab = np.array([[1, 2], [1, 2]], dtype=np.uint32)
        assert (cm.separate_touching_cells(lab, min_area=9) == lab).all()

    def test_missed_cell_restored_at_centroid(self):
        lab = cm.downsample_labels(_mask(), FAC, (0, 200, 0, 240))
        assert 3 not in lab
        ids = np.array([1, 3])
        rows = cm.to_map_index([40.0, 151.5], 0, FAC)
        cols = cm.to_map_index([40.0, 151.5], 0, FAC)
        restored, lost = cm.restore_missing_cells(lab, ids, rows, cols)
        assert (restored, lost) == (1, 0)
        assert lab[15, 15] == 3

    def test_colliding_centroids_keep_one_owner_and_count_the_rest_lost(self):
        lab = np.zeros((4, 4), dtype=np.uint32)
        lab[0, 0] = 9                                  # sampled cell holding pixel (0, 0)
        ids = np.array([5, 6, 7])
        rows, cols = np.array([2, 2, 0]), np.array([3, 3, 0])   # 5 and 6 share (2, 3); 7 hits cell 9
        restored, lost = cm.restore_missing_cells(lab, ids, rows, cols)
        assert (restored, lost) == (1, 2)
        assert lab[2, 3] == 5 and lab[0, 0] == 9

    def test_trailing_partial_block_is_sampled(self):
        m = np.zeros((203, 241), dtype=np.uint32)
        m[201:203, 30:40] = 7                          # lives only in the 3-row strip past row 200
        m[50:60, 240] = 8                              # lives only in the last column
        lab = cm.downsample_labels(m, FAC, (0, 203, 0, 241))
        assert lab.shape == (21, 25)                   # ceil(203/10), ceil(241/10)
        assert lab[20, 3] == 7 and lab[5, 24] == 8
        assert (cm.to_map_index([202, 240], 0, FAC) == [20, 24]).all()

    def test_wrong_mask_is_rejected(self, run_dir):
        spots = _spots()
        spots['Cell_ID'] = spots['Cell_ID'].map({1: 2, 2: 1, 3: 4})
        spots.to_csv(run_dir / f"{RUN}_processed" / "segmented" / "assigned_spots.csv", index=False)
        with pytest.raises(ValueError, match="not the mask"):
            _run(run_dir)


class TestOmePyramid:
    @pytest.fixture
    def small_levels(self, monkeypatch):
        """Force a 3-level pyramid and 16-px tiles (with ragged edge tiles) on the 20 x 24 map."""
        monkeypatch.setattr(cm, "PYRAMID_TOP_MAX", 8)
        monkeypatch.setattr(cm, "OME_TILE", 16)

    def test_default_output_is_named_channel_pyramid(self, run_dir, small_levels):
        stats = _run(run_dir)
        seg = run_dir / f"{RUN}_processed" / "segmented"
        levels, names, px, page = _read_ome(seg / "cellmap_Q20_genes.ome.tif")
        assert names == ['total', 'GeneA', 'GeneC', 'GeneB', 'GeneZ']
        assert px == pytest.approx(0.1625 * FAC)
        assert [lv.shape for lv in levels] == [(5, 20, 24), (5, 10, 12), (5, 5, 6)]
        assert stats['level_shapes'] == [(20, 24), (10, 12), (5, 6)]
        assert page.is_tiled and page.tilewidth == 16 and page.compression == 8   # deflate
        assert levels[0].dtype == np.uint8
        assert not (seg / "cellmap_Q20_total.zip").exists()   # total is channel 0 here

    def test_full_resolution_level_matches_the_imagej_maps(self, run_dir, small_levels):
        _run(run_dir)
        (base, *_), *_ = _read_ome(run_dir / f"{RUN}_processed" / "segmented" / "cellmap_Q20_genes.ome.tif")
        total, a, c, b, z = base
        assert (total[4, 4], total[4, 8], total[15, 15], total[12, 16]) == (4, 3, 4, 0)
        assert a[4, 4] == 3 and a[4, 8] == 1 and a[4, 6] == 0
        assert c[4, 8] == 2 and b[15, 15] == 4 and not z.any()

    def test_every_level_is_painted_from_the_mask(self, run_dir, small_levels):
        _run(run_dir)
        _, half, quarter = _read_ome(run_dir / f"{RUN}_processed" / "segmented" / "cellmap_Q20_genes.ome.tif")[0]
        # 2x level: cells 1 and 2 are 2 x 2 px -- too small for a border, so both stay whole
        assert half[1, 1, 2] == 3 and half[1, 1, 3] == 1       # GeneA of cell 1 | cell 2
        assert half[3, 7, 7] == 4                               # cell 3 restored at this level too
        assert set(np.unique(half)) <= {0, 1, 2, 3, 4}          # counts, never averaged
        assert quarter[0].max() == 4 and (quarter[3] == 4).sum() >= 1

    def test_roi_close_up(self, run_dir):
        stats = _run(run_dir, fac=2, roi=(10, 70, 10, 110), use_all=True, gene_file=None)
        seg = run_dir / f"{RUN}_processed" / "segmented"
        (base, *_), names, px, _ = _read_ome(seg / "cellmap_Q20_all_y10-70_x10-110.ome.tif")
        assert stats['map_shape'] == (30, 50) and base.shape == (4, 30, 50)
        assert names == ['total', 'GeneA', 'GeneB', 'GeneC']
        assert px == pytest.approx(0.325)
        assert base[1, 15, 15] == 3                   # full-res (40, 40) -> ROI map (15, 15)

    def test_fov_masked_run_is_named_and_counted_separately(self, run_dir):
        _run(run_dir, exclude_fov_masked=True)
        (base, *_), names, *_ = _read_ome(
            run_dir / f"{RUN}_processed" / "segmented" / "cellmap_Q20_fovmasked_genes.ome.tif")
        assert base[names.index('GeneC'), 4, 8] == 1  # one GeneC spot of cell 2 was fov-masked


class TestImageJOutputs:
    def test_stack_is_imagej_zip_with_gene_labels(self, run_dir):
        stats = _run(run_dir, fmt='imagej')
        seg = run_dir / f"{RUN}_processed" / "segmented"
        stack, meta, compression, xres, cmap = _read_zip(seg / "cellmap_Q20_genes.zip")
        assert stack.shape == (4, 20, 24) and stack.dtype == np.uint8
        assert compression == 1                        # uncompressed inside: ImageJ keeps labels
        assert meta['Labels'] == ['GeneA', 'GeneC', 'GeneB', 'GeneZ']
        assert meta['unit'] == 'um' and meta['min'] == 0
        assert xres[1] / xres[0] == pytest.approx(0.1625 * FAC)   # um per map pixel
        assert cmap is not None
        a, c, b, z = stack
        assert a[4, 4] == 3 and a[4, 8] == 1          # pixel value = per-cell count
        assert c[4, 8] == 2 and c[4, 4] == 0
        assert b[15, 15] == 4                          # the restored small cell
        assert a[4, 6] == 0                            # border between cells 1 and 2
        assert not z.any() and not stack[:, 12, 16].any()   # absent gene, cell without spots
        assert stats['genes_without_spots'] == ['GeneZ']
        assert stats['match_fraction'] == 1.0
        assert (stats['n_restored'], stats['n_lost']) == (1, 0)

    def test_total_map(self, run_dir):
        _run(run_dir, fmt='imagej')
        total, meta, *_ = _read_zip(run_dir / f"{RUN}_processed" / "segmented" / "cellmap_Q20_total.zip")
        assert total.shape == (20, 24)
        assert (total[4, 4], total[4, 8], total[15, 15], total[12, 16]) == (4, 3, 4, 0)
        assert meta['Labels'] in ('total', ['total'])   # tifffile unwraps a single label

    def test_fov_masked_total_is_named_to_match(self, run_dir):
        _run(run_dir, fmt='imagej', exclude_fov_masked=True)
        seg = run_dir / f"{RUN}_processed" / "segmented"
        stack, *_ = _read_zip(seg / "cellmap_Q20_fovmasked_genes.zip")
        assert stack[1, 4, 8] == 1
        assert (seg / "cellmap_Q20_fovmasked_total.zip").is_file()

    def test_plain_tif_output(self, run_dir):
        _run(run_dir, fmt='imagej', output="custom.tif")
        with tifffile.TiffFile(run_dir / f"{RUN}_processed" / "segmented" / "custom.tif") as tf:
            assert tf.imagej_metadata['Labels'][0] == 'GeneA'

    def test_over_imagej_limit_fails_before_writing(self, run_dir, monkeypatch):
        monkeypatch.setattr(cm, "IMAGEJ_MAX_BYTES", 1000)
        with pytest.raises(ValueError, match="4 GiB"):
            _run(run_dir, fmt='imagej')


class TestGeoJson:
    """Cell outlines + counts as QuPath measurements, in full-resolution mosaic pixels."""

    def _features(self, path):
        feats = json.loads(path.read_text(encoding="utf-8"))["features"]
        return {f["properties"]["measurements"]["Cell ID"]: f for f in feats}

    def test_every_cell_with_sparse_counts(self, run_dir):
        stats = _run(run_dir)
        feats = self._features(run_dir / f"{RUN}_processed" / "segmented" / "cellmap_Q20_genes.geojson")
        assert sorted(feats) == [1, 2, 3, 4] and stats['geojson_cells'] == 4
        m = {k: f["properties"]["measurements"] for k, f in feats.items()}
        assert m[1] == {"Cell ID": 1, "total": 4, "GeneA": 3, "GeneB": 1}
        assert m[2] == {"Cell ID": 2, "total": 3, "GeneA": 1, "GeneC": 2}
        assert m[4] == {"Cell ID": 4, "total": 0}        # segmented, no spots: still drawn
        for f in feats.values():
            assert f["properties"]["objectType"] == "detection"
            ring = f["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1] and len(ring) >= 4

    def test_outlines_sit_on_the_cells_in_full_res_pixels(self, run_dir):
        _run(run_dir)
        feats = self._features(run_dir / f"{RUN}_processed" / "segmented" / "cellmap_Q20_genes.geojson")
        xs, ys = zip(*feats[1]["geometry"]["coordinates"][0])
        assert 20 <= min(xs) and max(xs) <= 60 and 20 <= min(ys) and max(ys) <= 60   # cell 1
        xs, ys = zip(*feats[3]["geometry"]["coordinates"][0])                       # restored
        assert (min(xs), max(xs), min(ys), max(ys)) == (150, 160, 150, 160)
        xs, _ = zip(*feats[2]["geometry"]["coordinates"][0])
        assert min(xs) >= 60                           # cell 2 starts where cell 1 ends

    def test_roi_outlines_keep_mosaic_coordinates(self, run_dir):
        _run(run_dir, fac=2, roi=(10, 70, 10, 110), use_all=True, gene_file=None)
        feats = self._features(run_dir / f"{RUN}_processed" / "segmented"
                               / "cellmap_Q20_all_y10-70_x10-110.geojson")
        assert sorted(feats) == [1, 2]
        xs, ys = zip(*feats[1]["geometry"]["coordinates"][0])
        assert 20 <= min(xs) <= 22 and 58 <= max(xs) <= 60 and 20 <= min(ys) and max(ys) <= 60

    def test_one_pixel_necks_do_not_make_invalid_polygons(self):
        """Two blobs joined by a diagonal / one-pixel neck: the raw centre-line contour
        self-intersects, which makes QuPath refuse the WHOLE file."""
        shapely = pytest.importorskip("shapely")
        m = np.zeros((12, 12), np.uint32)
        m[1:5, 1:5] = 7
        m[5, 5] = 7                                     # diagonal chain to the second blob
        m[6:10, 6:10] = 7
        m[2, 8:11] = 7
        m[3, 10] = 7
        from scipy.ndimage import find_objects
        (sl,) = [s for s in find_objects(m) if s is not None]
        ring = cm._outline((m[sl] == 7).astype(np.uint8), sl, 10, (0, 0))
        assert shapely.Polygon(ring).is_valid and ring[0] == ring[-1]

    def test_can_be_switched_off(self, run_dir):
        _run(run_dir, geojson=False)
        assert not list((run_dir / f"{RUN}_processed" / "segmented").glob("*.geojson"))


class TestGuards:
    def test_no_spots_passing_threshold_fails(self, run_dir):
        with pytest.raises(ValueError, match="No spots"):
            cli.run_pipeline(RUN, gene_file=str(run_dir / "genes.txt"), threshold=1.0,
                             label="P1", fac=FAC, base_dir=str(run_dir))

    def test_unverifiable_pairing_warns(self, run_dir, caplog):
        _run(run_dir, roi=(160, 200, 0, 240))         # no spot inside this strip
        assert "Could not verify" in caplog.text

    def test_unknown_format_rejected(self, run_dir):
        with pytest.raises(ValueError, match="Unknown format"):
            cm.build_cell_maps(None, cm.filter_spots(_spots(), 0.99), ['GeneA'], "x", fac=FAC,
                               px_um=1.0, fmt='png')


class TestCli:
    def test_parse_roi(self):
        assert cli.parse_roi("100:200,300:450") == (100, 200, 300, 450)
        for bad in ("100:200", "200:100,0:5", "a:b,c:d"):
            with pytest.raises(argparse.ArgumentTypeError):
                cli.parse_roi(bad)

    def test_mask_resolution_prefers_cells_over_nuclei(self, run_dir):
        seg = run_dir / f"{RUN}_processed" / "segmented"
        tifffile.imwrite(seg / "nuclei_mask.tif", _mask())
        assert cli.resolve_mask(seg).name == "cellsam_mask.tif"
        assert cli.resolve_mask(seg, "nuclei_mask.tif").name == "nuclei_mask.tif"

    def test_subcommand_registered(self, monkeypatch, run_dir):
        from sprintseq.cli import main as main_mod
        seen = {}
        monkeypatch.setattr(cli, "run_pipeline", lambda run_id, **kw: seen.update(run_id=run_id, **kw))
        monkeypatch.setattr("sys.argv", ["sprintseq", "cell-map", "--run-id", RUN,
                                         "--gene-file", str(run_dir / "genes.txt"), "-Q", "30"])
        main_mod.main()
        assert seen['run_id'] == RUN and seen['label'] == "Q30"
        assert seen['fmt'] == 'ome' and seen['fac'] is None       # resolved per format downstream
        assert seen['threshold'] == pytest.approx(0.999)

    def test_default_resolution_follows_format(self):
        assert cli.DEFAULT_FAC == {'ome': 4, 'imagej': 10}

    def test_pyramid_stops_once_the_top_level_is_small(self):
        assert cm.pyramid_steps((10006, 12763)) == [1, 2, 4, 8, 16]
        assert cm.pyramid_steps((500, 800)) == [1]
