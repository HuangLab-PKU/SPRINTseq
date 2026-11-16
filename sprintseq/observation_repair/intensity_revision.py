"""Source-preserving application of independently replayed boundary readings."""
import numpy as np


def apply_verified_boundary_readings(stored, legacy_replay, full_support):
    """Correct only finite values that exactly replay the demonstrated old error.

    The caller must bind IDs, dimension order, image identity and measurement
    receipts before calling. This changes raw readings only, never observation
    validity or gene calls. Unexplained values are reported and left untouched.
    Missing acquisition values remain NaN even if an image crop contains zeros.
    """
    stored = np.asarray(stored)
    old = np.asarray(legacy_replay)
    new = np.asarray(full_support)
    if stored.ndim != 2 or not stored.shape[1] or old.shape != stored.shape or new.shape != stored.shape:
        raise ValueError('Matching nonempty observation-by-dimension arrays required')
    if not np.issubdtype(stored.dtype, np.floating):
        raise ValueError('Floating source required to preserve missing acquisition')
    if np.isinf(stored).any() or (stored[np.isfinite(stored)] < 0).any():
        raise ValueError('Invalid source intensity')
    if not np.isfinite(old).all() or not np.isfinite(new).all() or (old < 0).any() or (new < 0).any():
        raise ValueError('Replayed image readings must be finite and nonnegative')
    finite = np.isfinite(stored)
    needs = finite & (old != new) & (stored == old)
    already = finite & (stored == new)
    unexplained = finite & (stored != old) & (stored != new)
    result = stored.copy()
    result[needs] = new[needs]
    if not np.array_equal(result[needs], new[needs]):
        raise ValueError('Measured corrections are not exactly representable in source dtype')
    if not np.array_equal(np.isnan(result), np.isnan(stored)):
        raise ValueError('Missing acquisition changed')
    return {'values': result, 'needs_correction': needs, 'already_correct': already,
            'unexplained': unexplained, 'missing_preserved': ~finite}
