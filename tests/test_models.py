import pytest
import torch
from fewshot.models import PatchMemory, conformal_threshold, greedy_coreset, image_scores


def test_greedy_coreset_has_no_repeated_indices_even_for_duplicate_features():
    features = torch.zeros(7, 3)
    indices, radius = greedy_coreset(features, 7)
    assert len(indices.unique()) == 7 and radius == 0


def test_coverage_radius_does_not_increase_with_memory_budget():
    torch.manual_seed(3)
    features = torch.randn(80, 6)
    _, small = greedy_coreset(features, 4)
    _, large = greedy_coreset(features, 16)
    assert large <= small


def test_chunked_nearest_neighbor_maps_match_unchunked_maps():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    normal = torch.rand(3, 1, 32, 36)
    model = PatchMemory(16, chunk_size=7).fit(normal)
    a = model.score(normal[:1])
    model.chunk_size = 10000
    b = model.score(normal[:1])
    assert a.shape == normal[:1].shape and torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_conformal_order_statistic_and_insufficient_sample_case():
    scores = torch.arange(1, 21).float()
    assert conformal_threshold(scores, .1) == 19.
    assert conformal_threshold(scores, .01) == float("inf")
    # A tie at the threshold does not alarm under the strict decision rule.
    threshold = conformal_threshold(torch.ones(40), .05)
    assert not (torch.ones(1) > threshold).any()


def test_top_tail_image_score_is_monotone_under_added_anomaly_evidence():
    a = torch.zeros(1, 1, 20, 20)
    b = a.clone()
    b[:, :, :3, :3] = 5
    assert image_scores(b) > image_scores(a)


def test_scoring_requires_fitted_normal_memory():
    with pytest.raises(RuntimeError, match="Fit normal"):
        PatchMemory().score(torch.zeros(1, 1, 32, 32))
