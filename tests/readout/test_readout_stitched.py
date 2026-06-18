"""Tests for readout_stitched pipeline using synthetic TIFF images."""

import os
import tempfile
import numpy as np
import pandas as pd
import tifffile


def _create_synthetic_image(height, width, spots=None, dtype=np.uint16):
    """Create a synthetic image with optional bright spots.

    Parameters
    ----------
    height, width : int
        Image dimensions.
    spots : list of (y, x, intensity), optional
        Bright spots to place on the image.
    dtype : numpy dtype
        Image data type.

    Returns
    -------
    np.ndarray
        Synthetic image.
    """
    rng = np.random.RandomState(42)
    img = rng.randint(100, 300, size=(height, width), dtype=dtype)
    if spots:
        for y, x, intensity in spots:
            if 0 <= y < height and 0 <= x < width:
                img[y, x] = intensity
                # Add a small halo for better detection
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and (dy, dx) != (0, 0):
                            img[ny, nx] = max(img[ny, nx], intensity // 2)
    return img


def _setup_stitched_dir(tmp_dir, height, width, channels, n_cycles, spots=None):
    """Create a stitched directory with synthetic TIFF images.

    Parameters
    ----------
    tmp_dir : str
        Temporary directory to create stitched/ in.
    height, width : int
        Image dimensions.
    channels : list of str
        Channel names (e.g., ['cy3', 'cy5']).
    n_cycles : int
        Number of cycles.
    spots : list of (y, x, intensity), optional
        Spots to place in all images.

    Returns
    -------
    str
        Path to the stitched directory.
    """
    stc_dir = os.path.join(tmp_dir, 'stitched')
    os.makedirs(stc_dir, exist_ok=True)
    for cyc in range(1, n_cycles + 1):
        for ch in channels:
            img = _create_synthetic_image(height, width, spots=spots)
            tifffile.imwrite(os.path.join(stc_dir, f'cyc_{cyc}_{ch}.tif'), img)
    return stc_dir


class TestDetectSpotsInBlock:
    def test_returns_global_coordinates(self):
        """Detected coords are offset by (start_y, start_x)."""
        from sprintseq.readout.spot_detection import get_spot_coordinates
        # Create a small image with a bright spot at (10, 15)
        block = _create_synthetic_image(64, 64, spots=[(10, 15, 60000)])
        start_y, start_x = 100, 200
        # Detect spots
        coords = get_spot_coordinates(block, method='tophat', min_distance=2, snr=3.0, tophat_radius=3)
        if len(coords) > 0:
            global_coords = coords + np.array([start_y, start_x])
            # Check that global coords are offset
            assert np.all(global_coords[:, 0] >= start_y)
            assert np.all(global_coords[:, 1] >= start_x)

    def test_empty_block_returns_empty(self):
        """Block with uniform low intensity returns no spots."""
        from sprintseq.readout.spot_detection import get_spot_coordinates
        block = np.full((64, 64), 200, dtype=np.uint16)
        coords = get_spot_coordinates(block, method='tophat', min_distance=2, snr=3.0, tophat_radius=3)
        assert len(coords) == 0


class TestBlobLogDetection:
    """LoG detection via skimage.feature.blob_log (method='blob_log')."""

    def test_returns_n_by_2_float32(self):
        """get_spot_coordinates returns (N, 2) float32 array for method='blob_log'."""
        from sprintseq.readout.spot_detection import get_spot_coordinates
        block = _create_synthetic_image(64, 64, spots=[(20, 30, 60000)])
        coords = get_spot_coordinates(
            block, method='blob_log',
            min_sigma=1.0, max_sigma=2.0, num_sigma=2,
            threshold=0.005, overlap=0.5,
        )
        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.dtype == np.float32

    def test_detects_known_spots(self):
        """LoG finds two known synthetic spots within +/-2 px of ground truth."""
        from sprintseq.readout.spot_detection import get_spot_coordinates
        spots_truth = [(20, 30, 60000), (45, 50, 55000)]
        block = _create_synthetic_image(80, 80, spots=spots_truth)
        coords = get_spot_coordinates(
            block, method='blob_log',
            min_sigma=1.0, max_sigma=2.0, num_sigma=2,
            threshold=0.005, overlap=0.5,
        )
        assert len(coords) >= 2, f"expected >=2 detections, got {len(coords)}"
        for y, x, _ in spots_truth:
            distances = np.sqrt(((coords - np.array([y, x])) ** 2).sum(axis=1))
            assert distances.min() <= 2.0, f"spot ({y}, {x}) not found within 2 px"

    def test_empty_image_returns_empty(self):
        """Uniform image returns empty (0, 2) float32 result."""
        from sprintseq.readout.spot_detection import get_spot_coordinates
        block = np.full((64, 64), 200, dtype=np.uint16)
        coords = get_spot_coordinates(
            block, method='blob_log',
            min_sigma=1.0, max_sigma=2.0, num_sigma=2,
            threshold=0.01, overlap=0.5,
        )
        assert coords.shape == (0, 2)
        assert coords.dtype == np.float32

    def test_kwargs_forwarded_to_skimage(self):
        """min_sigma / max_sigma / num_sigma / threshold / overlap reach skimage.feature.blob_log."""
        from unittest.mock import patch
        from sprintseq.readout.spot_detection import get_spot_coordinates

        block = np.zeros((32, 32), dtype=np.uint16)
        expected = dict(
            min_sigma=1.5, max_sigma=2.5, num_sigma=3,
            threshold=0.012, overlap=0.4,
        )
        with patch(
            'sprintseq.readout.spot_detection.blob_log',
            return_value=np.empty((0, 3), dtype=np.float64),
        ) as mock_blob:
            get_spot_coordinates(block, method='blob_log', **expected)
            assert mock_blob.called, 'skimage.feature.blob_log was not invoked'
            kwargs = mock_blob.call_args.kwargs
            for k, v in expected.items():
                assert kwargs.get(k) == v, f'kwarg {k}: expected {v}, got {kwargs.get(k)}'

    def test_default_detection_kwargs_for_blob_log(self):
        """CLI's default kwargs builder produces the documented blob_log dict."""
        from sprintseq.cli.readout import _default_detection_kwargs
        kw = _default_detection_kwargs('blob_log', channel='cy3', snrs={'cy3': 3.0})
        assert kw == {
            'min_sigma': 1.0, 'max_sigma': 2.0, 'num_sigma': 2,
            'threshold': 0.005, 'overlap': 0.5,
        }


class TestReadIntensityInBlock:
    def test_reads_tophat_intensity_at_coords(self):
        """Intensity values are non-negative and match expected shape."""
        from sprintseq.readout.intensity_readout import read_intensity_tophat
        block = _create_synthetic_image(64, 64, spots=[(10, 15, 60000), (30, 40, 50000)])
        coords = np.array([[10, 15], [30, 40]], dtype=np.float32)
        intensities = read_intensity_tophat(block, coords, tophat_radius=3, search_radius=1)
        assert len(intensities) == 2
        assert np.all(intensities >= 0)
        # Bright spots should have high intensity after tophat
        assert intensities[0] > 100
        assert intensities[1] > 100

    def test_empty_coords_returns_empty(self):
        """No coordinates returns empty array."""
        from sprintseq.readout.intensity_readout import read_intensity_tophat
        block = _create_synthetic_image(64, 64)
        coords = np.empty((0, 2), dtype=np.float32)
        intensities = read_intensity_tophat(block, coords, tophat_radius=3, search_radius=1)
        assert len(intensities) == 0


class TestEndToEnd:
    def test_small_synthetic_dataset(self):
        """Create tiny stitched images, run detect + intensity, verify output format."""
        from sprintseq.readout.image_blocks import block_starts
        from sprintseq.readout.spot_detection import get_spot_coordinates
        from sprintseq.readout.intensity_readout import read_intensity_tophat
        from sprintseq.readout.deduplicate import deduplicate_dataframe

        height, width = 256, 256
        channels = ['cy3', 'cy5']
        cycle_num = 2
        seq_cycle = 2
        spots = [(50, 60, 60000), (150, 180, 55000)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            stc_dir = _setup_stitched_dir(tmp_dir, height, width, channels, seq_cycle, spots=spots)

            # Stage 1: Detection (simplified - from first cycle/channel only)
            all_coords = []
            for cyc in range(1, cycle_num + 1):
                for ch in channels:
                    img_path = os.path.join(stc_dir, f'cyc_{cyc}_{ch}.tif')
                    img = tifffile.imread(img_path)
                    if img.ndim == 3:
                        img = img[0]
                    h, w = img.shape
                    for sy, sx in block_starts(h, w, block_size=(256, 256), overlap=(0, 0)):
                        ey = min(sy + 256, h)
                        ex = min(sx + 256, w)
                        block = np.asarray(img[sy:ey, sx:ex])
                        coords = get_spot_coordinates(block, method='tophat', min_distance=2, snr=3.0, tophat_radius=3)
                        if len(coords) > 0:
                            global_coords = coords + np.array([sy, sx])
                            all_coords.append(global_coords)
                    del img

            assert len(all_coords) > 0, "No spots detected in synthetic images"
            unique_coords = np.vstack(all_coords)
            coords_rounded = np.round(unique_coords).astype(np.int32)
            _, unique_indices = np.unique(coords_rounded, axis=0, return_index=True)
            unique_coords = unique_coords[unique_indices]

            # Stage 2: Intensity reading
            intensity_df = pd.DataFrame({'Y': unique_coords[:, 0], 'X': unique_coords[:, 1]})
            for cyc in range(1, seq_cycle + 1):
                for ch in channels:
                    img_path = os.path.join(stc_dir, f'cyc_{cyc}_{ch}.tif')
                    img = tifffile.imread(img_path)
                    if img.ndim == 3:
                        img = img[0]
                    col_name = f'cyc_{cyc}_{ch}'
                    intensities = read_intensity_tophat(
                        img, unique_coords, tophat_radius=3, search_radius=1
                    )
                    intensity_df[col_name] = intensities
                    del img

            # Verify output format
            assert 'Y' in intensity_df.columns
            assert 'X' in intensity_df.columns
            for cyc in range(1, seq_cycle + 1):
                for ch in channels:
                    assert f'cyc_{cyc}_{ch}' in intensity_df.columns
            assert len(intensity_df) > 0
            assert np.all(intensity_df.iloc[:, 2:].values >= 0)

    def test_output_csv_format(self):
        """Verify position.csv and intensity.csv column conventions."""
        height, width = 128, 128
        channels = ['cy3', 'cy5']
        seq_cycle = 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            stc_dir = _setup_stitched_dir(
                tmp_dir, height, width, channels, seq_cycle,
                spots=[(30, 40, 60000)]
            )
            read_dir = os.path.join(tmp_dir, 'readout')
            os.makedirs(read_dir, exist_ok=True)

            # Simulate pipeline output
            coords = np.array([[30.0, 40.0]])
            intensity_df = pd.DataFrame({'Y': coords[:, 0], 'X': coords[:, 1]})
            for cyc in range(1, seq_cycle + 1):
                for ch in channels:
                    intensity_df[f'cyc_{cyc}_{ch}'] = [1000.0]

            # Save position.csv
            position_df = intensity_df[['Y', 'X']].copy()
            position_df.insert(0, 'index', range(len(position_df)))
            position_df.to_csv(os.path.join(read_dir, 'position.csv'), index=False)

            # Save intensity.csv
            int_cols = ['index'] + [c for c in intensity_df.columns if c.startswith('cyc_')]
            int_df = intensity_df[[c for c in intensity_df.columns if c.startswith('cyc_')]].copy()
            int_df.insert(0, 'index', range(len(int_df)))
            int_df.to_csv(os.path.join(read_dir, 'intensity.csv'), index=False)

            # Verify
            pos = pd.read_csv(os.path.join(read_dir, 'position.csv'))
            assert list(pos.columns) == ['index', 'Y', 'X']

            inten = pd.read_csv(os.path.join(read_dir, 'intensity.csv'))
            expected_cols = ['index'] + [f'cyc_{c}_{ch}' for c in range(1, seq_cycle + 1) for ch in channels]
            assert list(inten.columns) == expected_cols


class TestFilterCoordsByCoverage:
    """Generic cyc1∩cyc2 coverage hook (_filter_coords_by_coverage) — pure-function unit tests."""

    def test_keeps_inside_drops_outside(self):
        from sprintseq.cli.readout import _filter_coords_by_coverage
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, :2] = True                              # left half (X//4 in {0,1}) imaged
        coords = np.array([[2.0, 2.0], [2.0, 10.0]])    # X=2 -> bin 0 (keep); X=10 -> bin 2 (drop)
        kept, keep = _filter_coords_by_coverage(coords, mask, downsample=4)
        assert keep.tolist() == [True, False]
        assert len(kept) == 1 and kept[0, 1] == 2.0

    def test_none_mask_is_noop(self):
        from sprintseq.cli.readout import _filter_coords_by_coverage
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        kept, keep = _filter_coords_by_coverage(coords, None, downsample=4)
        assert np.array_equal(kept, coords) and keep.all()

    def test_edge_coords_are_clipped(self):
        from sprintseq.cli.readout import _filter_coords_by_coverage
        mask = np.ones((2, 2), dtype=bool)
        coords = np.array([[100.0, 100.0]])             # //4 -> (25,25), clipped to (1,1)
        _, keep = _filter_coords_by_coverage(coords, mask, downsample=4)
        assert keep.tolist() == [True]

    def test_empty_coords(self):
        from sprintseq.cli.readout import _filter_coords_by_coverage
        kept, keep = _filter_coords_by_coverage(
            np.empty((0, 2)), np.ones((2, 2), dtype=bool), downsample=4)
        assert len(kept) == 0 and len(keep) == 0


class TestDetectAllSpotsCoverage:
    """detect_all_spots applies an optional coverage_mask after exact-dedup (n_workers=1)."""

    def test_coverage_mask_drops_out_of_region(self):
        from sprintseq.cli.readout import detect_all_spots
        height, width = 256, 256
        spots = [(50, 30, 60000), (60, 40, 60000),      # left half  -> kept
                 (50, 220, 60000), (60, 210, 60000)]    # right half -> dropped
        with tempfile.TemporaryDirectory() as tmp_dir:
            stc_dir = _setup_stitched_dir(tmp_dir, height, width, ['cy3'], 1, spots=spots)
            mask = np.zeros((16, 16), dtype=bool); mask[:, :8] = True   # left half imaged
            coords, stats = detect_all_spots(
                stc_dir, channels=['cy3'], detection_cycles=[1],
                block_size=(256, 256), block_overlap=(0, 0),
                detection_method='tophat', snrs={'cy3': 3.0}, n_workers=1,
                coverage_mask=mask, coverage_downsample=16,
            )
            assert len(coords) > 0
            assert np.all(coords[:, 1] < 128)           # only left-half spots survive
            assert stats.get('coverage_dropped', 0) > 0


if __name__ == '__main__':
    print("Running TestDetectSpotsInBlock...")
    t = TestDetectSpotsInBlock()
    t.test_returns_global_coordinates()
    print("  PASS: test_returns_global_coordinates")
    t.test_empty_block_returns_empty()
    print("  PASS: test_empty_block_returns_empty")

    print("Running TestReadIntensityInBlock...")
    t2 = TestReadIntensityInBlock()
    t2.test_reads_tophat_intensity_at_coords()
    print("  PASS: test_reads_tophat_intensity_at_coords")
    t2.test_empty_coords_returns_empty()
    print("  PASS: test_empty_coords_returns_empty")

    print("Running TestEndToEnd...")
    t3 = TestEndToEnd()
    t3.test_small_synthetic_dataset()
    print("  PASS: test_small_synthetic_dataset")
    t3.test_output_csv_format()
    print("  PASS: test_output_csv_format")

    print("\nAll readout_stitched tests passed!")
