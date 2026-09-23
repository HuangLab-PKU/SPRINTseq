"""Transcript points as QuPath objects -- the spot layer under the per-cell gene maps.

WHY TWO SHAPES
--------------
A section carries 1-2 M Q20 spots. QuPath keeps every object in memory and hit-tests them on
each repaint, so a million individual point objects make panning unusable, while ONE
annotation per gene holding a ``MultiPoint`` ROI is ~10^2 objects for a whole section, draws
as fast as the image, and toggles per gene through the object's classification. That is the
default here (:func:`write_multipoint_geojson`).

When the per-spot detail is the point -- checking what a suspect codeword looks like in a
crop, or which cell a transcript fell in -- :func:`write_point_geojson` writes one detection
per transcript carrying its probability and ``Cell ID`` as measurements. That is for a gene
subset or an ROI; the CLI caps it and says so rather than producing a file that hangs QuPath.

Coordinates are full-resolution mosaic pixels ``(x, y)`` -- the same space as the stitched
image and the cell outlines from :mod:`sprintseq.segment.cellmap`, so the layers line up.
"""
import colorsys
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from sprintseq.cli.density import parse_gene_name

logger = logging.getLogger(__name__)

#: Spots whose "gene" is a decoding outcome, not a transcript.
NON_GENE = ('Background', 'Infeasible')
#: Columns a spot table must have; ``Probability`` / ``Cell_ID`` are used when present.
REQUIRED_COLUMNS = ('Y', 'X', 'Gene')
#: Decimals kept in the written coordinates: 0.1 px is far below the ~5 px spot spacing and
#: roughly halves the file against full float repr.
COORD_DECIMALS = 1


def gene_color(gene):
    """A stable, well-separated RGB for a gene name.

    Deterministic (a digest, not Python's salted ``hash``), so a gene keeps its colour across
    runs and sessions -- two sections of the same block can be compared by eye. Saturation and
    value are pinned high enough to stay visible on a dark fluorescence image.
    """
    digest = hashlib.blake2b(gene.encode("utf-8"), digest_size=4).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    sat = 0.55 + (digest[2] / 255.0) * 0.35
    val = 0.75 + (digest[3] / 255.0) * 0.25
    return [int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, sat, val)]


def load_spots(path, *, threshold=None, genes=None, roi=None, exclude_fov_masked=False):
    """Read a spot table and apply the same cuts the maps use.

    Accepts anything with ``Y``, ``X``, ``Gene`` -- ``segmented/assigned_spots.csv`` (which
    also has ``Cell_ID``) or a ``readout/`` position+mapping merge. Unlike the cell maps this
    does NOT require a cell: a transcript outside every cell is still a transcript, and
    leaving them out would misrepresent the tissue. ``roi`` is ``(y0, y1, x0, x1)`` in
    full-resolution pixels.
    """
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; has {list(header)}")
    usecols = [c for c in (*REQUIRED_COLUMNS, 'Probability', 'Cell_ID', 'fov_masked')
               if c in header]
    spots = pd.read_csv(path, usecols=usecols)
    n_read = len(spots)
    spots['Gene'] = spots['Gene'].map(parse_gene_name)
    keep = ~spots['Gene'].isin(NON_GENE)
    if threshold is not None:
        if 'Probability' not in spots.columns:
            raise ValueError(f"{path} has no Probability column; drop the threshold")
        keep &= spots['Probability'] > threshold
    if exclude_fov_masked:
        if 'fov_masked' not in spots.columns:
            raise ValueError(f"{path} has no fov_masked column (this run has no FOV mask)")
        keep &= ~spots['fov_masked'].astype(bool)
    if genes is not None:
        keep &= spots['Gene'].isin(list(genes))
    if roi is not None:
        y0, y1, x0, x1 = roi
        keep &= (spots['Y'] >= y0) & (spots['Y'] < y1) & (spots['X'] >= x0) & (spots['X'] < x1)
    spots = spots[keep]
    logger.info("Spots: %s rows in %s -> %s kept", f"{n_read:,}", getattr(path, 'name', path),
                f"{len(spots):,}")
    return spots.reset_index(drop=True)


def _coords(xs, ys):
    """``[[x, y], ...]`` as JSON text, without building a Python list of lists."""
    xf = np.round(np.asarray(xs, dtype=float), COORD_DECIMALS)
    yf = np.round(np.asarray(ys, dtype=float), COORD_DECIMALS)
    return "[" + ",".join(f"[{x:g},{y:g}]" for x, y in zip(xf, yf)) + "]"


def _feature(obj_type, geometry, name, gene, measurements, locked=True):
    props = {"objectType": obj_type, "classification": {"name": gene, "color": gene_color(gene)},
             "measurements": measurements}
    if name is not None:
        props["name"] = name
    if locked:
        props["isLocked"] = True
    return {"type": "Feature", "geometry": geometry, "properties": props}


def write_multipoint_geojson(spots, path, *, max_per_gene=None, seed=0):
    """One QuPath annotation per gene, holding every one of its spots as a MultiPoint.

    Written gene by gene so memory stays at one gene's coordinates. With ``max_per_gene`` a
    gene over the cap is sampled uniformly (seeded, so a re-run gives the same picture) and
    the feature records ``sampled fraction`` -- a density judged from a capped layer would
    otherwise be silently wrong. Returns a summary dict.
    """
    rng = np.random.default_rng(seed)
    genes = sorted(spots['Gene'].unique())
    written = sampled = 0
    per_gene = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"type":"FeatureCollection","features":[')
        for i, gene in enumerate(genes):
            sub = spots[spots['Gene'] == gene]
            n_all = len(sub)
            frac = 1.0
            if max_per_gene is not None and n_all > max_per_gene:
                sub = sub.iloc[np.sort(rng.choice(n_all, max_per_gene, replace=False))]
                frac = len(sub) / n_all
                sampled += 1
            measurements = {"count": int(n_all), "drawn": int(len(sub))}
            if frac < 1.0:
                measurements["sampled fraction"] = round(frac, 4)
            feature = _feature(
                "annotation", {"type": "MultiPoint", "coordinates": None},
                f"{gene} (n={n_all:,})" + ("" if frac == 1.0 else f", {frac:.0%} drawn"),
                gene, measurements)
            text = json.dumps(feature, separators=(",", ":"))
            # splice the coordinates in as text: json.dumps on a million-point list would
            # build the whole thing in memory twice
            text = text.replace('"coordinates":null', '"coordinates":' + _coords(sub['X'], sub['Y']))
            fh.write(("," if i else "") + text)
            written += len(sub)
            per_gene[gene] = int(n_all)
        fh.write("]}")
    return {"mode": "multipoint", "genes": len(genes), "spots": int(len(spots)),
            "drawn": written, "genes_sampled": sampled, "per_gene": per_gene}


def write_point_geojson(spots, path):
    """One QuPath detection per transcript, carrying its probability and cell.

    For a crop or a couple of genes: every object is individually selectable and its
    measurements show in the measurement table. Costly above ~10^5 objects, which is why the
    CLI caps it.
    """
    has_p = 'Probability' in spots.columns
    has_cell = 'Cell_ID' in spots.columns
    genes = sorted(spots['Gene'].unique())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"type":"FeatureCollection","features":[')
        for i, row in enumerate(spots.itertuples(index=False)):
            measurements = {}
            if has_p:
                measurements["Probability"] = round(float(row.Probability), 4)
            if has_cell:
                measurements["Cell ID"] = int(row.Cell_ID)
            geometry = {"type": "Point",
                        "coordinates": [round(float(row.X), COORD_DECIMALS),
                                        round(float(row.Y), COORD_DECIMALS)]}
            fh.write(("," if i else "")
                     + json.dumps(_feature("detection", geometry, None, row.Gene, measurements,
                                           locked=False), separators=(",", ":")))
        fh.write("]}")
    return {"mode": "per-spot", "genes": len(genes), "spots": int(len(spots)),
            "drawn": int(len(spots)), "genes_sampled": 0,
            "per_gene": spots['Gene'].value_counts().to_dict()}
