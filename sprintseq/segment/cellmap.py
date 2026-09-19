"""Per-cell gene maps: every segmented cell painted with its transcript count.

The cell-level counterpart of the per-gene density maps (``sprintseq density``).
Density bins spots into fixed ``fac x fac`` squares; here each cell's own mask
footprint carries that cell's count for one gene, so the map shows real cell
positions and shapes, and the pixel value under the cursor in ImageJ *is* the
cell's count.

Steps:

1. The full-resolution label mask is downsampled by stride sampling -- the
   centre pixel of every ``fac x fac`` block -- through a Zarr view of the TIFF,
   so tiles are decoded as they are needed instead of loading an 8 GB uint32 mask.
2. Touching cells big enough to spare a pixel get a 1-px zero border, so
   neighbours with equal counts stay distinguishable (the cell-outline look of
   Xenium Explorer, in raster form).
3. Cells that no sample point hit (smaller than a block) are restored as one
   pixel at their spot centroid, so no cell with transcripts drops out.
4. Per-cell counts (postcode ``P > threshold``, the same cut as density) are
   painted through a label -> count lookup table, one slice per gene.

Two output formats:

- **Pyramidal OME-TIFF** (``write_ome_pyramid``, the default): tiled, deflate-
  compressed BigTIFF, one channel per gene plus a ``total`` channel, with
  sub-resolution levels in SubIFDs. QuPath and Fiji's Bio-Formats importer read
  only the tiles and level on screen, so memory does not scale with the file.
  Every level is painted from its own label map (steps 2-3 redone at that
  resolution) rather than resampled from the level above, so counts stay
  integers and every cell stays whole at every zoom.
- **ImageJ-ZIP** (``write_cell_map``): an uncompressed ImageJ TIFF inside a
  ``.zip`` -- ImageJ's own "Save As > ZIP" format, opened natively with the
  thermal LUT and display range preset, but loaded whole into memory. The TIFF
  must stay uncompressed: ImageJ opens a compressed multi-page TIFF through
  ``ij.io.Opener.openTiffStack``, which adds every slice with a null label.
  Zipping keeps the labels and shrinks the mostly-zero maps ~50-100x.

uint8 is written whenever every value fits.
"""

import json
import logging
import math
import os
import tempfile
import zipfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tifffile

from sprintseq.cli.density import parse_gene_name
from sprintseq.cli.density_stack import thermal_colormap

logger = logging.getLogger(__name__)

NON_GENE = ('Background', 'Infeasible')
SPOT_COLUMNS = ['Y', 'X', 'Gene', 'Probability', 'Cell_ID']
# Classic (non-Big) TIFF tops out at 4 GiB and ImageJ stacks cannot be BigTIFF;
# keep headroom for the header, labels and LUT.
IMAGEJ_MAX_BYTES = 2**32 - 2**26
DEFAULT_MIN_BORDER_AREA = 9   # map pixels; smaller cells keep every pixel
CHECK_BAND_ROWS = 1024        # one tile row of a cellSAM mask
MATCH_ERROR = 0.5             # below this, mask and spot table are different segmentations
MATCH_WARN = 0.95
OME_TILE = 512
PYRAMID_TOP_MAX = 1024        # add 2x levels until the coarsest fits in this many px
CELL_CLASS_COLOR = [220, 220, 220]   # QuPath class colour of the GeoJSON cells (grey)


@contextmanager
def open_label_array(mask_path):
    """Yield a lazily-read 2-D Zarr view of a label-mask TIFF (tiled or striped)."""
    import zarr

    with tifffile.TiffFile(mask_path) as tf:
        shape = tf.series[0].shape
        if len(shape) != 2:
            raise ValueError(f"Expected a 2-D label mask, got shape {shape}: {mask_path}")
        store = tf.aszarr()
        try:
            yield zarr.open(store, mode='r')
        finally:
            store.close()


def clip_roi(shape, fac, roi=None):
    """Clip a full-resolution ROI ``(y0, y1, x0, x1)`` (half-open) to *shape*."""
    h, w = shape
    y0, y1, x0, x1 = (0, h, 0, w) if roi is None else roi
    y0, x0, y1, x1 = max(0, y0), max(0, x0), min(h, y1), min(w, x1)
    if y1 - y0 < fac or x1 - x0 < fac:
        raise ValueError(f"ROI {(y0, y1, x0, x1)} is smaller than one {fac} x {fac} block "
                         f"(mask shape {shape}).")
    return y0, y1, x0, x1


def downsample_labels(labels, fac, roi):
    """Stride-sample *labels*: the centre pixel of every ``fac x fac`` block of *roi*.

    Map pixel ``(i, j)`` covers full-resolution block
    ``[y0 + i*fac, y0 + (i+1)*fac) x [x0 + j*fac, x0 + (j+1)*fac)``, so the map
    has ``ceil(height / fac)`` rows. A trailing partial block too short to hold
    its centre is sampled at its last row / column instead of being dropped.
    """
    y0, y1, x0, x1 = roi
    off = fac // 2
    n_rows, n_cols = -(-(y1 - y0) // fac), -(-(x1 - x0) // fac)
    lab = np.asarray(labels[y0 + off:y1:fac, x0 + off:x1:fac])
    if lab.shape[0] < n_rows:
        lab = np.vstack([lab, np.asarray(labels[y1 - 1:y1, x0 + off:x1:fac])])
    if lab.shape[1] < n_cols:
        edge = np.asarray(labels[y0 + off:y1:fac, x1 - 1:x1])
        if edge.shape[0] < n_rows:
            edge = np.vstack([edge, np.asarray(labels[y1 - 1:y1, x1 - 1:x1])])
        lab = np.hstack([lab, edge])
    return lab


def to_map_index(coord, origin, fac):
    """Full-resolution coordinate(s) -> map row/col (floor of the block index)."""
    return np.floor((np.asarray(coord, dtype=np.float64) - origin) / fac).astype(np.int64)


def check_mask_matches_spots(labels, spots, roi, band_rows=CHECK_BAND_ROWS):
    """Check that the spot table's ``Cell_ID`` values came from this mask.

    Reads one full-resolution band (the one holding the most spots) and compares
    the label under each spot -- at ``int(Y), int(X)``, as ``assign_spots_to_cells``
    does -- with its ``Cell_ID``. Only spots on a labelled pixel count: a nucleus
    KD-tree assignment legitimately places spots outside any mask object.

    Returns ``(agreement, n_checked)``; agreement is NaN when nothing could be checked.
    """
    y0, y1, x0, x1 = roi
    yi = spots['Y'].to_numpy().astype(np.int64)
    xi = spots['X'].to_numpy().astype(np.int64)
    in_roi = (yi >= y0) & (yi < y1) & (xi >= x0) & (xi < x1)
    if not in_roi.any():
        return float('nan'), 0
    band_idx = (yi[in_roi] - y0) // band_rows
    b0 = y0 + int(np.bincount(band_idx).argmax()) * band_rows
    b1 = min(y1, b0 + band_rows)
    sel = in_roi & (yi >= b0) & (yi < b1)
    band = np.asarray(labels[b0:b1, x0:x1])
    under = band[yi[sel] - b0, xi[sel] - x0]
    on_cell = under > 0
    n = int(on_cell.sum())
    if n == 0:
        return float('nan'), 0
    cell_ids = spots['Cell_ID'].to_numpy()[sel][on_cell]
    return float(np.mean(under[on_cell] == cell_ids)), n


def filter_spots(spots, threshold, exclude_fov_masked=False):
    """Keep gene-assigned spots with ``Probability > threshold`` (density's cut).

    Gene names are normalised with ``parse_gene_name`` so they match density TIFF
    names and gene files. With *exclude_fov_masked*, spots flagged ``fov_masked``
    (decode-side tile mask) are dropped, as for ``cell_gene_matrix_fovmasked.csv``.
    """
    missing = [c for c in SPOT_COLUMNS if c not in spots.columns]
    if missing:
        raise ValueError(f"Spot table lacks columns {missing}; has {list(spots.columns)}")
    keep = (~spots['Gene'].isin(NON_GENE)) & (spots['Probability'] > threshold) & (spots['Cell_ID'] > 0)
    if exclude_fov_masked:
        if 'fov_masked' not in spots.columns:
            raise ValueError("--exclude-fov-masked needs a 'fov_masked' column in the spot table.")
        keep &= ~spots['fov_masked'].astype(bool)
    out = spots.loc[keep, SPOT_COLUMNS].copy()
    out['Gene'] = out['Gene'].astype(str).map(parse_gene_name)
    return out


def cell_gene_counts(spots, genes):
    """Per-cell counts of *genes*, plus per-cell totals and spot centroids.

    Returns ``(cell_ids, counts, totals, cy, cx)``: ``counts`` is
    ``(n_cells, len(genes))``; ``totals`` sums every gene in *spots*, not only
    *genes*; ``cy``/``cx`` are the mean spot position of each cell (full-res px).
    """
    codes, cell_ids = pd.factorize(spots['Cell_ID'].to_numpy(), sort=True)
    n_cells, n_genes = len(cell_ids), len(genes)
    totals = np.bincount(codes, minlength=n_cells)
    cy = np.bincount(codes, weights=spots['Y'].to_numpy(), minlength=n_cells) / totals
    cx = np.bincount(codes, weights=spots['X'].to_numpy(), minlength=n_cells) / totals
    gcode = spots['Gene'].map({g: i for i, g in enumerate(genes)}).to_numpy()
    sel = ~pd.isna(gcode)
    flat = codes[sel].astype(np.int64) * n_genes + gcode[sel].astype(np.int64)
    counts = np.bincount(flat, minlength=n_cells * n_genes).reshape(n_cells, n_genes)
    return np.asarray(cell_ids, dtype=np.int64), counts, totals, cy, cx


def separate_touching_cells(lab, min_area=DEFAULT_MIN_BORDER_AREA):
    """Zero a 1-px line wherever two different cells touch.

    A pixel is zeroed when the pixel above or to its left belongs to another
    cell, so each contact costs one pixel on one side only. Cells smaller than
    *min_area* map pixels are left whole -- a border would erase most of them.
    """
    edge = np.zeros(lab.shape, dtype=bool)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[:-1, :] != 0)
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, :-1] != 0)
    edge &= lab != 0
    if min_area > 1:
        area = np.bincount(lab.ravel())
        edge &= area[lab] >= min_area
    out = lab.copy()
    out[edge] = 0
    return out


def restore_missing_cells(lab, cell_ids, rows, cols):
    """Paint one pixel at the centroid of every cell that sampling missed.

    Only cells whose centroid lies on the map are considered. A centroid pixel
    already taken -- by a sampled cell, or by another missed cell whose centroid
    falls in the same pixel -- keeps its first owner; the others count as lost.
    Modifies *lab* in place and returns ``(n_restored, n_lost)``.
    """
    h, w = lab.shape
    size = max(int(lab.max()), int(cell_ids.max(initial=0))) + 1
    present = np.bincount(lab.ravel(), minlength=size) > 0
    on_map = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    missing = on_map & ~present[cell_ids]
    r, c, ids = rows[missing], cols[missing], cell_ids[missing]
    free = lab[r, c] == 0
    _, first = np.unique(r[free] * w + c[free], return_index=True)
    lab[r[free][first], c[free][first]] = ids[free][first]
    return len(first), len(ids) - len(first)


def auto_display_max(values):
    """99th percentile of the non-zero values (at least 1): a starting B&C range."""
    nz = values[values > 0]
    return float(max(1.0, np.percentile(nz, 99))) if nz.size else 1.0


def smallest_dtype(values):
    return np.uint8 if values.size == 0 or values.max() <= 255 else np.uint16


def write_cell_map(path, lab, cell_ids, values, slice_labels, *, px_um,
                   display_range, dtype=None):
    """Paint ``values[:, k]`` onto the label map and write an ImageJ stack.

    *values* is ``(n_cells, n_slices)``, row-aligned with *cell_ids*. A ``.zip``
    *path* gets the ImageJ-ZIP format (the TIFF is staged in the local temp dir,
    then deflated into the archive); any other suffix is written as a plain
    ImageJ TIFF. Returns the uncompressed TIFF size in bytes.
    """
    path = str(path)
    if not path.lower().endswith('.zip'):
        return _write_imagej_tiff(path, lab, cell_ids, values, slice_labels,
                                  px_um=px_um, display_range=display_range, dtype=dtype)
    arcname = os.path.splitext(os.path.basename(path))[0] + '.tif'
    with tempfile.TemporaryDirectory(prefix='cellmap_') as tmp:
        tif = os.path.join(tmp, arcname)
        nbytes = _write_imagej_tiff(tif, lab, cell_ids, values, slice_labels,
                                    px_um=px_um, display_range=display_range, dtype=dtype)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(tif, arcname=arcname)
    return nbytes


def _write_imagej_tiff(path, lab, cell_ids, values, slice_labels, *, px_um,
                       display_range, dtype=None):
    """Write the painted pages one at a time, so only one slice is in memory."""
    values = np.asarray(values)
    n = values.shape[1]
    dtype = np.dtype(dtype or smallest_dtype(values))
    nbytes = n * lab.size * dtype.itemsize
    if nbytes > IMAGEJ_MAX_BYTES:
        raise ValueError(
            f"{n} slices of {lab.shape[0]} x {lab.shape[1]} {dtype} = {nbytes / 2**30:.1f} GiB, "
            f"over the 4 GiB an ImageJ TIFF can hold. Raise --fac, narrow --roi, "
            f"or split the gene file.")
    lut = np.zeros(max(int(lab.max()), int(cell_ids.max(initial=0))) + 1, dtype=dtype)
    top = np.iinfo(dtype).max

    def pages():
        for k in range(n):
            lut[:] = 0
            lut[cell_ids] = np.minimum(values[:, k], top)
            yield lut[lab]

    metadata = {
        'axes': 'ZYX' if n > 1 else 'YX',
        'min': float(display_range[0]),
        'max': float(display_range[1]),
        'unit': 'um',
        'loop': False,
        'Labels': list(slice_labels),
        'Properties': {'CurrentLUT': 'Thermal (edited)'},
    }
    shape = (n, *lab.shape) if n > 1 else lab.shape
    tifffile.imwrite(
        path, pages(), shape=shape, dtype=dtype,
        imagej=True, photometric='minisblack', colormap=thermal_colormap(),
        resolution=(1.0 / px_um, 1.0 / px_um), metadata=metadata,
    )
    return nbytes


def pyramid_steps(shape, top_max=None):
    """Downsample steps 1, 2, 4, ... until the coarsest level fits in *top_max* px."""
    top_max = top_max or PYRAMID_TOP_MAX
    steps = [1]
    while max(math.ceil(n / steps[-1]) for n in shape) > top_max:
        steps.append(steps[-1] * 2)
    return steps


def render_label_map(lab_raw, step, *, fac, origin, cell_ids, cy, cx,
                     min_border_area=DEFAULT_MIN_BORDER_AREA):
    """The label map of one pyramid level, drawn from the unbordered base map.

    Resamples *lab_raw* (base map at ``fac``) by *step*, then adds borders and
    restores missed cells at that level's own resolution. *lab_raw* is not
    modified. Returns ``(lab, n_restored, n_lost)``.
    """
    h, w = lab_raw.shape
    lab = lab_raw if step == 1 else downsample_labels(lab_raw, step, (0, h, 0, w))
    lab = separate_touching_cells(lab, min_area=min_border_area) if min_border_area > 0 else lab.copy()
    y0, x0 = origin
    restored, lost = restore_missing_cells(lab, cell_ids, to_map_index(cy, y0, fac * step),
                                           to_map_index(cx, x0, fac * step))
    return lab, restored, lost


def write_ome_pyramid(path, label_maps, steps, cell_ids, values, channel_names, *,
                      px_um, tile=None, dtype=None):
    """Paint ``values[:, k]`` onto every level and write a pyramidal OME-TIFF.

    ``label_maps[i]`` is the map at downsample ``steps[i]`` (``steps[0] == 1``,
    full resolution = *px_um*). Levels 1.. go into SubIFDs of each base plane,
    the layout Bio-Formats and QuPath read as one multi-resolution image. Tiles
    are painted one channel at a time and encoded as they are yielded, so only
    one painted plane is in memory. Returns the file size in bytes.
    """
    tile = tile or OME_TILE
    values = np.asarray(values)
    n = values.shape[1]
    dtype = np.dtype(dtype or smallest_dtype(values))
    top = np.iinfo(dtype).max
    size = max(max(int(m.max()) for m in label_maps), int(cell_ids.max(initial=0))) + 1
    lut = np.zeros(size, dtype=dtype)

    def tiles(lab):
        h, w = lab.shape
        for k in range(n):
            lut[:] = 0
            lut[cell_ids] = np.minimum(values[:, k], top)
            plane = lut[lab]
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    yield plane[y:y + tile, x:x + tile]

    options = dict(dtype=dtype, tile=(tile, tile), compression='zlib',
                   photometric='minisblack', resolutionunit='CENTIMETER')
    metadata = {
        'axes': 'CYX',
        'Channel': {'Name': list(channel_names)},
        'PhysicalSizeX': px_um, 'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': px_um, 'PhysicalSizeYUnit': 'µm',
    }
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tif:
        for i, (step, lab) in enumerate(zip(steps, label_maps)):
            res = 1e4 / (px_um * step)   # pixels per cm
            level = (dict(subifds=len(steps) - 1, metadata=metadata) if i == 0
                     else dict(subfiletype=1))
            tif.write(tiles(lab), shape=(n, *lab.shape), resolution=(res, res),
                      **level, **options)
    return os.path.getsize(path)


def _outline(sub, sl, fac, origin):
    """Largest outer contour of one cell's bbox mask, as a closed full-res ``(x, y)`` ring.

    Vertices come from the label map at ``fac`` (pixel centres, +/- fac/2 full-res px);
    a cell too thin for a contour (1-2 map px, or a line) gets its pixel-edge bounding box.
    """
    import cv2

    y0, x0 = origin
    oy, ox = sl[0].start, sl[1].start
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :] if cnts else None
    if cnt is None or len(cnt) < 3 or cv2.contourArea(cnt) == 0:
        ys, xs = np.nonzero(sub)
        c0, c1, r0, r1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        pts = [(x0 + (ox + c) * fac, y0 + (oy + r) * fac)
               for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1))]
    else:
        pts = [(x0 + (ox + c + 0.5) * fac, y0 + (oy + r + 0.5) * fac) for c, r in cnt]
    ring = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
    return _valid_ring(ring + [ring[0]])


def _valid_ring(ring):
    """Return a ring QuPath will accept.

    A contour through pixel centres folds back on itself across a one-pixel neck, and a
    single self-intersecting polygon makes QuPath's GeoJSON import fail outright
    ("Reduction failed, possible invalid input"; 277 of 181k BZ07 cells). Invalid rings are
    repaired with shapely (largest part of ``make_valid``), falling back to the convex hull;
    without shapely installed, every ring is replaced by its hull (valid by construction).
    """
    try:
        import shapely
    except ImportError:
        import cv2

        hull = cv2.convexHull(np.asarray(ring[:-1], np.float32))[:, 0, :].tolist()
        return [[round(x, 1), round(y, 1)] for x, y in hull] + [[round(hull[0][0], 1),
                                                                 round(hull[0][1], 1)]]
    poly = shapely.Polygon(ring)
    if poly.is_valid:
        return ring
    fixed = shapely.make_valid(poly)
    parts = [g for g in shapely.get_parts(fixed) if g.geom_type == "Polygon" and g.area > 0]
    candidates = ([max(parts, key=lambda g: g.area)] if parts else []) + [poly.convex_hull]
    for cand in candidates:
        out = [[round(x, 1), round(y, 1)] for x, y in cand.exterior.coords]
        if len(out) >= 4 and shapely.Polygon(out).is_valid:
            return out
    return ring


def write_cells_geojson(path, lab, cell_ids, counts, totals, genes, *, fac, origin,
                        extra_pixels=()):
    """Every cell in ``lab`` as a QuPath detection, with its counts as measurements.

    Coordinates are full-resolution mosaic pixels (QuPath's space for the stitched image).
    Measurements: ``Cell ID``, ``total`` (all genes) and each gene with a NON-ZERO count --
    QuPath reads an absent measurement as NaN, so in a measurement map a cell without the
    gene keeps its grey class colour rather than the bottom of the colour scale (the
    Xenium look), and the file stays small. Cells without any spot are included too
    (``total`` 0), so every segmented cell shows. ``extra_pixels`` = ``[(label, row, col)]``
    adds cells present only as a restored single map pixel. Returns the feature count.
    """
    from scipy.ndimage import find_objects

    row_of = {int(c): i for i, c in enumerate(cell_ids)}

    def feature(label, ring):
        i = row_of.get(label)
        meas = {"Cell ID": label, "total": int(totals[i]) if i is not None else 0}
        if i is not None:
            meas.update({genes[k]: int(counts[i, k]) for k in np.flatnonzero(counts[i])})
        return json.dumps({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"objectType": "detection",
                           "classification": {"name": "Cell", "color": CELL_CLASS_COLOR},
                           "measurements": meas, "isLocked": True},
        }, separators=(",", ":"))

    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"type":"FeatureCollection","features":[')
        for label, sl in enumerate(find_objects(lab), start=1):
            if sl is None:
                continue
            sub = (lab[sl] == label).astype(np.uint8)
            fh.write(("," if n else "") + feature(label, _outline(sub, sl, fac, origin)))
            n += 1
        for label, r, c in extra_pixels:
            sl = (slice(int(r), int(r) + 1), slice(int(c), int(c) + 1))
            fh.write(("," if n else "") + feature(int(label), _outline(
                np.ones((1, 1), np.uint8), sl, fac, origin)))
            n += 1
        fh.write("]}")
    return n


def build_cell_maps(mask_path, spots, genes, output_path, *, fac, px_um, roi=None,
                    fmt='ome', total_path=None, min_border_area=DEFAULT_MIN_BORDER_AREA,
                    display_max=None, total_display_max=None, geojson_path=None):
    """Build the per-cell gene maps for one mask.

    ``fmt='ome'``: one pyramidal OME-TIFF at *output_path*, channels
    ``['total', *genes]``. ``fmt='imagej'``: the gene stack at *output_path* and
    the total-count map at *total_path*, each with an ImageJ display range
    (*display_max* / *total_display_max*, else the 99th percentile).

    *spots* must already be filtered (``filter_spots``). Returns a dict of
    summary statistics for logging and tests.
    """
    if fmt not in ('ome', 'imagej'):
        raise ValueError(f"Unknown format {fmt!r}; choose 'ome' or 'imagej'.")
    if fmt == 'imagej' and total_path is None:
        raise ValueError("The imagej format writes the total map separately: pass total_path.")
    if spots.empty:
        raise ValueError("No spots left after filtering -- check the threshold and the spot table.")
    stats = {}
    with open_label_array(mask_path) as labels:
        roi = clip_roi(labels.shape, fac, roi)
        agreement, n_checked = check_mask_matches_spots(labels, spots, roi)
        stats.update(mask_shape=tuple(labels.shape), roi=roi,
                     match_fraction=agreement, match_checked=n_checked)
        if not n_checked:
            logger.warning("Could not verify that %s matches the spot table: no spot in view "
                           "sits on a labelled pixel. Check --mask / --spots / --roi.", mask_path)
        if n_checked and agreement < MATCH_ERROR:
            raise ValueError(
                f"Only {agreement:.1%} of {n_checked:,} checked spots carry the Cell_ID of the "
                f"mask label under them: {mask_path} is not the mask this spot table was "
                f"assigned with. Pass the matching --mask / --spots pair.")
        if n_checked and agreement < MATCH_WARN:
            logger.warning("Mask/spot Cell_ID agreement is %.1f%% (%d spots checked); expected ~100%% "
                           "for mask-based assignment.", 100 * agreement, n_checked)
        lab_raw = downsample_labels(labels, fac, roi)

    y0, _, x0, _ = roi
    cell_ids, counts, totals, cy, cx = cell_gene_counts(spots, genes)
    steps = pyramid_steps(lab_raw.shape) if fmt == 'ome' else [1]
    maps, n_restored, n_lost = [], 0, 0
    for step in steps:
        lab, restored, lost = render_label_map(
            lab_raw, step, fac=fac, origin=(y0, x0), cell_ids=cell_ids, cy=cy, cx=cx,
            min_border_area=min_border_area)
        maps.append(lab)
        if step == 1:
            n_restored, n_lost = restored, lost
    base = maps[0]

    rows, cols = to_map_index(cy, y0, fac), to_map_index(cx, x0, fac)
    in_view = (rows >= 0) & (rows < base.shape[0]) & (cols >= 0) & (cols < base.shape[1])
    size = max(int(cell_ids.max(initial=0)), int(base.max()), int(lab_raw.max())) + 1
    present = np.bincount(base.ravel(), minlength=size) > 0
    shown = present[cell_ids]

    if geojson_path is not None:
        # Outlines from the UNBORDERED map (the 1-px separation lines would shave a side off
        # every cell); cells only restored as a centroid pixel are added as that pixel.
        in_raw = np.bincount(lab_raw.ravel(), minlength=size) > 0
        only_restored = shown & ~in_raw[cell_ids] & in_view
        extra = [(int(i), int(r), int(c)) for i, r, c in
                 zip(cell_ids[only_restored], rows[only_restored], cols[only_restored])
                 if base[r, c] == i]
        stats["geojson_cells"] = write_cells_geojson(
            geojson_path, lab_raw, cell_ids, counts, totals, genes, fac=fac, origin=(y0, x0),
            extra_pixels=extra)
    del lab_raw
    stats.update(
        map_shape=base.shape, level_shapes=[m.shape for m in maps],
        n_cells=len(cell_ids), n_cells_in_view=int(in_view.sum()), n_cells_shown=int(shown.sum()),
        n_restored=n_restored, n_lost=n_lost,
        genes_without_spots=[g for g, c in zip(genes, counts.sum(axis=0)) if c == 0],
    )

    if fmt == 'imagej':
        gene_max = display_max if display_max is not None else auto_display_max(counts[shown])
        total_max = (total_display_max if total_display_max is not None
                     else auto_display_max(totals[shown]))
        stats.update(
            display_max=gene_max, total_display_max=total_max,
            stack_bytes=write_cell_map(output_path, base, cell_ids, counts, genes,
                                       px_um=px_um, display_range=(0, gene_max)),
            total_bytes=write_cell_map(total_path, base, cell_ids, totals[:, None], ['total'],
                                       px_um=px_um, display_range=(0, total_max)),
        )
    else:
        values = np.column_stack([totals, counts])
        itemsize = np.dtype(smallest_dtype(values)).itemsize
        stats.update(
            raw_bytes=sum(m.size for m in maps) * values.shape[1] * itemsize,
            file_bytes=write_ome_pyramid(output_path, maps, steps, cell_ids, values,
                                         ['total', *genes], px_um=px_um),
        )
    return stats
