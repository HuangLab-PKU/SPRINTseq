"""Transcript points for QuPath (`sprintseq spot-map`).

What has to hold for the file to be usable: full-resolution mosaic coordinates in (x, y) --
the same frame as the image and the cell outlines, so the layers line up; one MultiPoint
annotation per gene by default, because a million individual objects make QuPath crawl; a
stable per-gene colour so two sections read the same; spots OUTSIDE cells kept (they are
transcripts, not noise); and a per-spot layer that refuses to be built at whole-section size
instead of producing a file that hangs the viewer.
"""
import argparse
import json

import pandas as pd
import pytest

from sprintseq.cli import spotmap as cli
from sprintseq.readout import spot_geojson as sg

RUN = "20990101_TEST_spotmap"


def _spots():
    rows = [
        (100.25, 200.75, 'GeneA', 1.0, 5),
        (110.0, 210.0, 'GeneA', 0.995, 0),        # outside every cell -- still a transcript
        (120.0, 220.0, 'GeneA', 0.5, 7),          # below Q20
        (130.0, 230.0, 'SP_12_GeneC', 1.0, 8),    # prefix stripped, as in density / cell-map
        (140.0, 240.0, 'Background', 1.0, 9),     # not a gene
        (150.0, 250.0, 'GeneB', 0.999, 3),
        (9000.0, 9000.0, 'GeneB', 1.0, 4),        # far away: outside the test ROI
    ]
    df = pd.DataFrame(rows, columns=['Y', 'X', 'Gene', 'Probability', 'Cell_ID'])
    df['fov_masked'] = [False, False, False, True, False, False, False]
    return df


@pytest.fixture
def run_dir(tmp_path):
    seg = tmp_path / f"{RUN}_processed" / "segmented"
    seg.mkdir(parents=True)
    _spots().to_csv(seg / "assigned_spots.csv", index=False)
    return tmp_path


def _run(run_dir, **kw):
    kw.setdefault('use_all', True)
    kw.setdefault('label', 'Q20')
    return cli.run_pipeline(RUN, base_dir=str(run_dir), **kw)


def _features(run_dir, name):
    path = run_dir / f"{RUN}_processed" / "segmented" / name
    return json.loads(path.read_text(encoding="utf-8"))["features"], path


def test_one_multipoint_annotation_per_gene_in_image_coordinates(run_dir):
    stats = _run(run_dir)
    feats, path = _features(run_dir, "spots_Q20.geojson")
    assert path.name in stats['path']
    by_gene = {f["properties"]["classification"]["name"]: f for f in feats}
    assert sorted(by_gene) == ["GeneA", "GeneB", "GeneC"]      # Background + the Q20 cut gone
    a = by_gene["GeneA"]
    assert a["geometry"]["type"] == "MultiPoint"
    assert a["properties"]["objectType"] == "annotation"
    # (x, y) in full-resolution pixels: X first, Y second -- swapping them mirrors the layer
    assert a["geometry"]["coordinates"] == [[200.8, 100.2], [210.0, 110.0]]
    assert a["properties"]["measurements"] == {"count": 2, "drawn": 2}
    assert "GeneA" in a["properties"]["name"]
    assert stats["spots"] == 5 and stats["drawn"] == 5 and stats["genes"] == 3


def test_a_spot_outside_every_cell_is_kept(run_dir):
    """cell-map drops Cell_ID == 0 by design; the spot layer must not -- those transcripts
    are real and leaving them out would misrepresent the tissue."""
    _run(run_dir)
    feats, _ = _features(run_dir, "spots_Q20.geojson")
    coords = {tuple(c) for f in feats for c in f["geometry"]["coordinates"]}
    assert (210.0, 110.0) in coords


def test_gene_colour_is_stable_and_visible():
    assert sg.gene_color("CD3E") == sg.gene_color("CD3E")
    assert sg.gene_color("CD3E") != sg.gene_color("KRT19")
    for gene in ("CD3E", "KRT19", "GeneA", "clonotype12"):
        assert max(sg.gene_color(gene)) >= 140      # never a near-black class colour


def test_gene_and_roi_filters(run_dir):
    _run(run_dir, use_all=False, genes="GeneB")
    feats, _ = _features(run_dir, "spots_Q20.geojson")
    assert [f["properties"]["classification"]["name"] for f in feats] == ["GeneB"]
    assert len(feats[0]["geometry"]["coordinates"]) == 2

    _run(run_dir, use_all=False, genes="GeneB", roi=(0, 1000, 0, 1000))
    feats, _ = _features(run_dir, "spots_Q20_y0-1000_x0-1000.geojson")
    assert feats[0]["geometry"]["coordinates"] == [[250.0, 150.0]]


def test_exclude_fov_masked_matches_the_maps(run_dir):
    stats = _run(run_dir, exclude_fov_masked=True)
    feats, _ = _features(run_dir, "spots_Q20_fovmasked.geojson")
    assert "GeneC" not in {f["properties"]["classification"]["name"] for f in feats}
    assert stats["spots"] == 4                      # the one fov-masked GeneC spot is gone


def test_sampling_records_the_fraction_it_drew(run_dir):
    stats = _run(run_dir, max_per_gene=1)
    feats, _ = _features(run_dir, "spots_Q20.geojson")
    a = next(f for f in feats if f["properties"]["classification"]["name"] == "GeneA")
    m = a["properties"]["measurements"]
    assert m == {"count": 2, "drawn": 1, "sampled fraction": 0.5}
    assert len(a["geometry"]["coordinates"]) == 1
    assert stats["genes_sampled"] == 2 and stats["drawn"] == 3


def test_per_spot_layer_carries_probability_and_cell(run_dir):
    _run(run_dir, per_spot=True)
    feats, _ = _features(run_dir, "spots_Q20_perspot.geojson")
    assert len(feats) == 5 and all(f["geometry"]["type"] == "Point" for f in feats)
    assert all(f["properties"]["objectType"] == "detection" for f in feats)
    one = next(f for f in feats if f["geometry"]["coordinates"] == [200.8, 100.2])
    assert one["properties"]["measurements"] == {"Probability": 1.0, "Cell ID": 5}
    assert one["properties"]["classification"]["name"] == "GeneA"


def test_per_spot_refuses_a_whole_section_unless_forced(run_dir, monkeypatch):
    monkeypatch.setattr(cli, "PER_SPOT_LIMIT", 2)
    with pytest.raises(ValueError, match="crawl"):
        _run(run_dir, per_spot=True)
    assert not (run_dir / f"{RUN}_processed" / "segmented" / "spots_Q20_perspot.geojson").exists()
    _run(run_dir, per_spot=True, force=True)
    feats, _ = _features(run_dir, "spots_Q20_perspot.geojson")
    assert len(feats) == 5


def test_falls_back_to_the_readout_tables_when_unsegmented(tmp_path):
    read = tmp_path / f"{RUN}_processed" / "readout"
    read.mkdir(parents=True)
    spots = _spots()
    spots[['Y', 'X']].to_csv(read / "position.csv", index_label="index")
    spots[['Gene', 'Probability']].to_csv(read / "mapping_postcode.csv", index_label="index")
    stats = cli.run_pipeline(RUN, base_dir=str(tmp_path), use_all=True, label='Q20')
    assert (read / "spots_Q20.geojson").is_file()
    assert stats["spots"] == 5
    assert not (read / ".spot_map_merged.csv").exists()      # the temp join is cleaned up
    feats = json.loads((read / "spots_Q20.geojson").read_text(encoding="utf-8"))["features"]
    assert "Cell ID" not in json.dumps(feats)                # no segmentation -> no cell ids


def test_cli_wires_the_flags(run_dir, monkeypatch):
    from sprintseq.cli import main as main_mod

    seen = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda run_id, **kw: seen.update(kw, run=run_id))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    main_mod._build_spot_map(sub)
    args = parser.parse_args(["spot-map", "--run-id", RUN, "--genes", "GeneA", "--per-spot",
                              "--roi", "0:10,0:10", "-Q", "30", "--exclude-fov-masked"])
    args._func(args)
    assert seen["run"] == RUN and seen["genes"] == "GeneA" and seen["per_spot"] is True
    assert seen["roi"] == (0, 10, 0, 10) and seen["label"] == "Q30"
    assert seen["threshold"] == pytest.approx(0.999) and seen["exclude_fov_masked"] is True
