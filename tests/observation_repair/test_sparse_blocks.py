import numpy as np
from sprintseq.observation_repair.discovery import block_intersects_region, sample


def test_skipped_blocks_cannot_contain_accepted_integer_spots():
    rng = np.random.default_rng(19)
    shape = (103, 127)
    mask = rng.random((13, 17)) > .98
    for sy in range(0, shape[0], 16):
        for sx in range(0, shape[1], 16):
            ey, ex = min(sy + 16, shape[0]), min(sx + 16, shape[1])
            yy, xx = np.mgrid[sy:ey, sx:ex]
            truth = sample(mask, np.c_[yy.ravel(), xx.ravel()], shape).any()
            assert bool(block_intersects_region(mask, shape, sy, sx, ey, ex)) == bool(truth)
