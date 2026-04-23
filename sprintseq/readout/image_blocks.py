"""
Block grid utilities for large-image processing.

block_starts() yields (start_y, start_x) for each block in a grid; the main
readout script uses this to open memmaps and read blocks per task.
"""

from typing import Tuple, List


def block_starts(
    height: int,
    width: int,
    block_size: Tuple[int, int] = (2048, 2048),
    overlap: Tuple[int, int] = (64, 64),
) -> List[Tuple[int, int]]:
    """Return list of (start_y, start_x) for each block covering a grid.

    Parameters
    ----------
    height, width : int
        Image shape (H, W).
    block_size : tuple (int, int)
        (by, bx) block size.
    overlap : tuple (int, int)
        (oy, ox) overlap between adjacent blocks.

    Returns
    -------
    list of (int, int)
        [(start_y, start_x), ...] for each block.
    """
    by, bx = block_size
    oy, ox = overlap
    step_y = max(1, by - oy)
    step_x = max(1, bx - ox)
    out: List[Tuple[int, int]] = []
    for start_y in range(0, height, step_y):
        for start_x in range(0, width, step_x):
            out.append((start_y, start_x))
    return out
