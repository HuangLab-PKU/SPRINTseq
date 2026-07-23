"""Per-method detection defaults must actually be accepted by that method.

`_default_detection_kwargs` fell through to `{'snr', 'tophat_radius'}` for every
non-spotiflow, non-blob_log method. `get_spot_coordinates` pops `snr`, but passes the
rest to the feature extractor -- and `feature_dog` / `feature_gaussian_dog` take
(sigma[, sigma1, sigma2, normalize_percentile]) with no `tophat_radius`. So
`DETECTION_METHOD = 'dog'` raised TypeError on the first block, i.e. two of the six
advertised methods could not run at all.
"""
import inspect

import numpy as np
import pytest

from sprintseq.cli import readout as ro
from sprintseq.readout import get_spot_coordinates
from sprintseq.readout import spot_detection as sd


def _image():
    rng = np.random.default_rng(0)
    img = rng.integers(80, 140, size=(96, 96)).astype(np.uint16)
    for y, x in ((20, 20), (40, 65), (70, 30)):
        img[y - 1:y + 2, x - 1:x + 2] += 1200
    return img


TRADITIONAL = ['dog', 'gaussian_dog', 'tophat', 'gaussian_tophat']


@pytest.mark.parametrize("method", TRADITIONAL)
def test_default_kwargs_are_accepted_by_the_feature_extractor(method):
    """The defaults must match the signature of the function they reach."""
    kw = dict(ro._default_detection_kwargs(method, 'cy3', ro.SNRS))
    kw.pop('snr', None)  # get_spot_coordinates pops this before dispatching
    fn = {
        'dog': sd.feature_dog,
        'gaussian_dog': sd.feature_gaussian_dog,
        'tophat': sd.feature_tophat,
        'gaussian_tophat': sd.feature_gaussian_tophat,
    }[method]
    accepted = set(inspect.signature(fn).parameters) - {'image'}
    assert set(kw) <= accepted, (
        f"{method}: default kwargs {sorted(set(kw) - accepted)} are not accepted by "
        f"{fn.__name__}{inspect.signature(fn)}")


@pytest.mark.parametrize("method", TRADITIONAL)
def test_every_advertised_method_actually_runs(method):
    """End-to-end guard: the defaults the pipeline would really use must not raise."""
    kw = ro._default_detection_kwargs(method, 'cy3', ro.SNRS)
    coords = get_spot_coordinates(_image(), method=method, min_distance=2, **kw)
    assert coords.ndim == 2 and coords.shape[1] == 2


def test_snr_still_reaches_the_threshold_for_every_method():
    """snr is the only cross-method knob; it must not be dropped by the fix."""
    for method in TRADITIONAL:
        assert 'snr' in ro._default_detection_kwargs(method, 'cy3', ro.SNRS)
