"""Deduplication for removing duplicate spots (KDTree + connected components).

Logic aligned with PRISM readout.utils.deduplicate: coordinates within a
Chebyshev distance threshold are grouped, one representative per group kept
(optionally the one with highest score).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def deduplicate_coordinates(coordinates, threshold=2, scores=None):
    """Remove duplicate coordinates within a threshold distance.

    If multiple coordinates are within the threshold distance, keep the one
    with the highest score (if scores provided) or the first one.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates
    threshold : float
        Distance threshold for considering coordinates as duplicates (default: 2 pixels)
    scores : np.ndarray, optional
        (N,) array of scores for each coordinate. Higher scores are preferred.
        If None, first coordinate in each group is kept.

    Returns
    -------
    tuple
        (kept_indices, deduplicated_coordinates)
        kept_indices: array of indices of kept coordinates
        deduplicated_coordinates: (M, 2) array of deduplicated coordinates
    """
    if len(coordinates) == 0:
        return np.array([], dtype=int), np.empty((0, 2), dtype=coordinates.dtype)

    coords = np.asarray(coordinates)
    coords_orig = coords

    # When scores given, sort by score descending so "first" in each component = best
    order = None
    if scores is not None:
        scores_arr = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores_arr)
        coords = coords[order]

    # Build KDTree for efficient distance queries
    tree = cKDTree(coords)

    # Find pairs within threshold (Chebyshev distance = box region)
    pairs = tree.query_pairs(r=threshold, p=float('inf'))

    if len(pairs) == 0:
        return np.arange(len(coords_orig)), np.asarray(coords_orig)

    # Find connected components (groups of duplicates)
    n_points = len(coords)
    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    data = np.ones(len(pairs), dtype=bool)
    adj = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    n_components, labels = connected_components(adj, directed=False)

    # One representative per component: smallest index (first in array order)
    min_idx_per_label = np.full(n_components, n_points)
    for idx in range(n_points):
        lab = labels[idx]
        min_idx_per_label[lab] = min(min_idx_per_label[lab], idx)
    kept_indices = min_idx_per_label

    if order is not None:
        kept_indices = order[kept_indices]
    deduplicated_coords = coords_orig[kept_indices]

    return kept_indices, deduplicated_coords


def deduplicate_dataframe(df, coordinate_columns=('Y', 'X'), threshold=2, sort_by=None):
    """Remove duplicate rows based on coordinate proximity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing coordinate columns and other data
    coordinate_columns : list or tuple
        Column names for coordinates (default: ('Y', 'X'))
    threshold : float
        Distance threshold in pixels, Chebyshev (default: 2)
    sort_by : str, optional
        Column name to use as score; higher values are preferred.
        If None, first row in each group is kept.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame
    """
    if len(df) == 0:
        return df.copy()

    coordinates = df[list(coordinate_columns)].values

    scores = None
    if sort_by is not None and sort_by in df.columns:
        scores = pd.to_numeric(df[sort_by], errors='coerce').to_numpy(dtype=np.float64)

    kept_indices, _ = deduplicate_coordinates(
        coordinates, threshold=threshold, scores=scores
    )

    return df.iloc[kept_indices].copy()
