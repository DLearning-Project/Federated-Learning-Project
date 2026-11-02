"""Simulator page backend: partition, synthetic adapters, round simulation, endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.language_models.core.partition import dirichlet_label_counts
from app.modules.language_models.core.simulator import client_rank, sample_clients, simulate_round
from app.modules.language_models.core.synthetic import M, N, synthetic_adapters

client = TestClient(app)
API = "/api/v1/language-models"
ROUND = dict(total_clients=10, clients_per_round=3, round=1, alpha=0.5, heterogeneous=False, tau=0.9, seed=42, bytes_per_param=4)


# --- Partition --------------------------------------------------------------------------


def test_partition_column_sums_and_determinism():
    a = dirichlet_label_counts(20, 16, 60, alpha=0.5, seed=1)
    b = dirichlet_label_counts(20, 16, 60, alpha=0.5, seed=1)
    assert a.shape == (20, 16)
    np.testing.assert_array_equal(a, b)
    assert np.all(a.sum(axis=0) == 60) and np.all(a >= 0)


def test_small_alpha_is_more_skewed_than_large_alpha():
    def mean_top_class_share(alpha):
        counts = dirichlet_label_counts(20, 16, 60, alpha=alpha, seed=3).astype(float)
        totals = counts.sum(axis=1)
        keep = totals > 0
        return float(np.mean(counts[keep].max(axis=1) / totals[keep]))

    assert mean_top_class_share(0.05) > 0.5 > mean_top_class_share(100.0)


def test_partition_rejects_bad_alpha():
    with pytest.raises(ValueError):
        dirichlet_label_counts(5, 4, 10, alpha=0, seed=0)


# --- Synthetic adapters ---------------------------------------------------------------------


def test_synthetic_adapters_shapes_ranks_and_determinism():
    Bs, As = synthetic_adapters(7, [0, 3, 9], [1, 2, 4])
    for B, A, r in zip(Bs, As, [1, 2, 4]):
        assert B.shape == (M, r) and A.shape == (r, N)
        assert np.linalg.matrix_rank(B @ A) == r
    Bs2, _ = synthetic_adapters(7, [0, 3, 9], [1, 2, 4])
    np.testing.assert_array_equal(Bs[0], Bs2[0])


# --- Round simulation -------------------------------------------------------------------


def test_sampling_is_seeded_and_changes_with_round():
    assert sample_clients(10, 3, 1, 42) == sample_clients(10, 3, 1, 42)
    rounds = {tuple(sample_clients(10, 3, t, 42)) for t in range(1, 8)}
    assert len(rounds) > 1


def test_simulate_round_homogeneous():
    r = simulate_round(10, 3, 1, alpha=0.5, heterogeneous=False, tau=0.9, seed=42, bytes_per_param=4)
    assert r.ranks == [4, 4, 4]
    assert r.weights.sum() == pytest.approx(1.0)
    assert r.steps.aggregation_error["fedit"] is not None
    assert r.steps.aggregation_error["exact_stack"].absolute < 1e-9
    assert set(r.payload.methods) == {"fedit", "exact_stack", "florist"}
    assert r.payload.methods["florist"].download_rank == r.steps.p


def test_simulate_round_heterogeneous_notices():
    r = simulate_round(10, 10, 1, alpha=0.5, heterogeneous=True, tau=0.9, seed=42, bytes_per_param=4)
    assert sorted(set(r.ranks)) == [1, 2, 4]
    assert r.steps.aggregation_error["fedit"] is None
    assert "fedit" not in r.payload.methods
    assert any("FedIT is not computed" in n for n in r.notices)
    assert any("compact-computation advantage" in n for n in r.notices)  # sum of ranks 19 >= 16


def test_client_rank_pattern():
    assert [client_rank(i, True) for i in range(10)] == [1, 1, 1, 1, 1, 2, 2, 2, 4, 4]
    assert client_rank(3, False) == 4


# --- Endpoints ----------------------------------------------------------------------------


def test_round_endpoint():
    res = client.post(f"{API}/simulator/round", json=ROUND)
    assert res.status_code == 200
    body = res.json()
    r = body["results"]
    assert body["inputs"] == ROUND
    assert r["provenance"] == "computed_live_synthetic"
    assert (r["m"], r["n"]) == (16, 32)
    assert len(r["class_counts"]) == 10 and len(r["class_counts"][0]) == 16
    assert 1 <= r["p"] <= len(r["singular_values"])
    assert r["cumulative_energy"][-1] == pytest.approx(1.0)
    assert r["reconstruction_error"]["absolute"] == pytest.approx(r["eq5_bound"], abs=1e-9)
    assert {p["method"] for p in r["payload"]} == {"fedit", "exact_stack", "florist"}
    assert any("SYNTHETIC" in a for a in body["assumptions"])


@pytest.mark.parametrize(
    "change",
    [{"clients_per_round": 11}, {"total_clients": 51}, {"tau": 0}, {"alpha": 0}, {"bytes_per_param": 3}, {"extra": 1}],
)
def test_round_endpoint_rejects_invalid(change):
    assert client.post(f"{API}/simulator/round", json={**ROUND, **change}).status_code == 422


def test_round_endpoint_rejects_more_sampled_than_total():
    res = client.post(f"{API}/simulator/round", json={**ROUND, "total_clients": 4, "clients_per_round": 5})
    assert res.status_code == 422
    assert "at most total_clients = 4" in res.json()["detail"]


def test_glossary_and_paper_endpoints():
    terms = client.get(f"{API}/glossary").json()
    assert len(terms) == 26
    assert all(t["equation_plain"] and t["levels"]["beginner"] for t in terms)
    paper = client.get(f"{API}/paper").json()
    assert paper["provenance"] == "paper_reported"
    assert len(paper["reported_results"]["table3"]["rows"]) == 7
