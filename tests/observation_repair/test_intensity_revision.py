import numpy as np
import pytest
from sprintseq.observation_repair.intensity_revision import apply_verified_boundary_readings


def test_reproduced_boundary_and_missing_acquisition_are_separate():
    stored = np.array([[1024., np.nan, 12., 17., 8.]], dtype=np.float32)
    old = np.array([[1024., 0., 10., 13., 8.]])
    new = np.array([[1028., 0., 12., 19., 8.]])
    result = apply_verified_boundary_readings(stored, old, new)
    np.testing.assert_array_equal(result['values'], [[1028., np.nan, 12., 17., 8.]])
    np.testing.assert_array_equal(result['needs_correction'], [[True, False, False, False, False]])
    np.testing.assert_array_equal(result['unexplained'], [[False, False, False, True, False]])
    assert np.isnan(stored[0, 1]) and stored[0, 0] == 1024
    assert result['values'].dtype == stored.dtype


@pytest.mark.parametrize('bad', [np.inf, -1.])
def test_invalid_source_is_rejected(bad):
    with pytest.raises(ValueError, match='source intensity'):
        apply_verified_boundary_readings(np.array([[bad]]), np.zeros((1, 1)), np.ones((1, 1)))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.])
def test_invalid_measurement_cannot_be_written(bad):
    with pytest.raises(ValueError, match='Replayed'):
        apply_verified_boundary_readings(np.zeros((1, 1)), np.zeros((1, 1)), np.array([[bad]]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match='Matching'):
        apply_verified_boundary_readings(np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((2, 2)))


def test_rounding_correction_to_source_dtype_is_rejected():
    with pytest.raises(ValueError, match='representable'):
        apply_verified_boundary_readings(np.array([[1.]], dtype=np.float32), np.array([[1.]]), np.array([[1.0000000001]]))
