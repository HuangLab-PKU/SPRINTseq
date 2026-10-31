"""Boundary contracts for the stable PoSTcode posterior normalizer."""
import pytest

torch = pytest.importorskip("torch")
from torch.distributions import MultivariateNormal
from sprintseq.gene_calling.gene_mapping import _postcode_e_step_stable


def fixture():
    theta = torch.tensor([[0., 0.], [2., 2.]])
    return theta, torch.eye(2), torch.tensor([.3, .7])


def test_matches_linear_space_for_well_conditioned_data():
    theta, sigma, weights = fixture()
    data = torch.tensor([[0., 0.], [1., -1.]])
    linear = torch.stack([weights[k] * MultivariateNormal(theta[k], sigma).log_prob(data).exp() for k in range(2)], 1)
    expected = linear / linear.sum(1, keepdim=True)
    actual = _postcode_e_step_stable(data, weights, theta, sigma, 2, 2)
    assert torch.allclose(actual, expected.double(), atol=2e-7, rtol=2e-6)


def test_underflow_retains_finite_normalized_posterior():
    theta, sigma, weights = fixture()
    data = torch.tensor([[1000., 1000.], [-1000., -1000.]])
    linear = torch.stack([weights[k] * MultivariateNormal(theta[k], sigma).log_prob(data).exp() for k in range(2)], 1)
    assert (linear == 0).all()
    actual = _postcode_e_step_stable(data, weights, theta, sigma, 2, 2)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.sum(1), torch.ones(2, dtype=torch.float64))
    assert torch.equal(actual.argmax(1), torch.tensor([1, 0]))


def test_zero_weight_receives_no_posterior_mass():
    theta, sigma, _ = fixture()
    actual = _postcode_e_step_stable(torch.zeros(1, 2), torch.tensor([0., 1.]), theta, sigma, 1, 2)
    assert torch.equal(actual, torch.tensor([[0., 1.]], dtype=torch.float64))


def test_class_permutation_preserves_predictions():
    theta, sigma, weights = fixture()
    data = torch.tensor([[.5, 1.]])
    actual = _postcode_e_step_stable(data, weights, theta, sigma, 1, 2)
    permuted = _postcode_e_step_stable(data, weights.flip(0), theta.flip(0), sigma, 1, 2)
    assert torch.allclose(actual, permuted.flip(1))


def test_no_supported_class_fails_instead_of_fabricating_probabilities():
    theta, sigma, _ = fixture()
    with pytest.raises(RuntimeError, match="no finite class likelihood"):
        _postcode_e_step_stable(torch.zeros(1, 2), torch.zeros(2), theta, sigma, 1, 2)
