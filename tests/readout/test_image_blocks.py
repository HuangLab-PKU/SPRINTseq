"""Tests for block_starts utility."""

from sprintseq.readout.image_blocks import block_starts


class TestBlockStarts:
    def test_single_block_smaller_than_block_size(self):
        """Image smaller than block size yields one block at (0,0)."""
        result = block_starts(100, 200, block_size=(2048, 2048), overlap=(64, 64))
        assert result == [(0, 0)]

    def test_exact_fit_no_overlap(self):
        """Image exactly divisible by block_size with overlap=(0,0)."""
        result = block_starts(4096, 4096, block_size=(2048, 2048), overlap=(0, 0))
        assert len(result) == 4
        assert (0, 0) in result
        assert (0, 2048) in result
        assert (2048, 0) in result
        assert (2048, 2048) in result

    def test_blocks_cover_full_image(self):
        """Every pixel in image is covered by at least one block."""
        height, width = 5000, 6000
        bs = (2048, 2048)
        ov = (64, 64)
        starts = block_starts(height, width, block_size=bs, overlap=ov)

        by, bx = bs
        for py in range(0, height, 500):  # sample pixels
            for px in range(0, width, 500):
                covered = False
                for sy, sx in starts:
                    if sy <= py < min(sy + by, height) and sx <= px < min(sx + bx, width):
                        covered = True
                        break
                assert covered, f"Pixel ({py}, {px}) not covered by any block"

    def test_overlap_produces_more_blocks(self):
        """Overlap > 0 produces more blocks than overlap = 0."""
        no_overlap = block_starts(4096, 4096, block_size=(2048, 2048), overlap=(0, 0))
        with_overlap = block_starts(4096, 4096, block_size=(2048, 2048), overlap=(64, 64))
        assert len(with_overlap) > len(no_overlap)

    def test_step_calculation(self):
        """step = block_size - overlap; blocks start at multiples of step."""
        bs = (2048, 2048)
        ov = (64, 64)
        step_y = bs[0] - ov[0]  # 1984
        step_x = bs[1] - ov[1]  # 1984
        result = block_starts(10000, 10000, block_size=bs, overlap=ov)
        for sy, sx in result:
            assert sy % step_y == 0, f"start_y={sy} not multiple of step_y={step_y}"
            assert sx % step_x == 0, f"start_x={sx} not multiple of step_x={step_x}"

    def test_first_block_is_origin(self):
        """First block always starts at (0, 0)."""
        result = block_starts(5000, 5000, block_size=(2048, 2048), overlap=(64, 64))
        assert result[0] == (0, 0)

    def test_zero_dimension_yields_empty(self):
        """height=0 or width=0 yields empty list."""
        assert block_starts(0, 100, block_size=(2048, 2048), overlap=(64, 64)) == []
        assert block_starts(100, 0, block_size=(2048, 2048), overlap=(64, 64)) == []


if __name__ == '__main__':
    t = TestBlockStarts()
    for name in dir(t):
        if name.startswith('test_'):
            getattr(t, name)()
            print(f"  PASS: {name}")
    print("\nAll block_starts tests passed!")
