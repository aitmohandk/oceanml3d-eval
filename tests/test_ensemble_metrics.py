"""Probabilistic scores (ported from fm-opencode evaluation/metrics.py) on synthetic ensembles."""
import numpy as np

from oceanml3d_eval.metrics.ensemble import (
    EnsembleScores,
    crps_ensemble,
    energy_score,
    rank_histogram,
    spread_skill,
)


def test_crps_of_one_member_is_mae():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((4, 5))
    member = truth + 0.3
    assert abs(crps_ensemble(member[None], truth) - 0.3) < 1e-9


def test_crps_rewards_a_calibrated_ensemble():
    rng = np.random.default_rng(1)
    truth = np.zeros((200,))
    good = rng.normal(0, 1, (50, 200))          # centred on the truth
    biased = rng.normal(2, 1, (50, 200))        # same spread, wrong centre
    overconfident = rng.normal(0, 0.05, (50, 200))
    assert crps_ensemble(good, truth) < crps_ensemble(biased, truth)
    assert crps_ensemble(overconfident, truth) < crps_ensemble(good, truth)   # sharp AND right wins


def test_energy_score_penalises_bias():
    rng = np.random.default_rng(2)
    truth = rng.standard_normal((3, 4))
    ens = truth[None] + rng.normal(0, 0.1, (20, 3, 4))
    assert energy_score(ens, truth) < energy_score(ens + 1.0, truth)


def test_spread_skill_ratio_is_one_for_a_calibrated_ensemble():
    """Calibrated = truth and members are exchangeable draws around a common mean."""
    rng = np.random.default_rng(3)
    mu = rng.standard_normal((5000,))
    truth = mu + rng.normal(0, 1.0, 5000)
    ens = mu[None] + rng.normal(0, 1.0, (40, 5000))
    spread, rmse, ratio = spread_skill(ens, truth)
    assert 0.9 < ratio < 1.1 and 0.9 < spread < 1.1
    _, _, under = spread_skill(mu[None] + rng.normal(0, 0.1, (40, 5000)), truth)
    assert under < 0.3                                    # over-confident ensemble
    _, _, over = spread_skill(mu[None] + rng.normal(0, 5.0, (40, 5000)), truth)
    assert over > 2.0                                     # over-dispersive ensemble


def test_rank_histogram_is_flat_when_calibrated():
    rng = np.random.default_rng(4)
    truth = rng.standard_normal((4000,))
    ens = rng.standard_normal((9, 4000))                  # truth and members from the same law
    counts = rank_histogram(ens, truth)
    assert len(counts) == 10
    assert counts.std() / counts.mean() < 0.15
    biased = rank_histogram(ens + 3, truth)
    assert biased[0] > 0.8 * biased.sum()                 # truth always below the members


def test_metric_on_an_ensemble_product(products, truth_field, tmp_path):
    import xarray as xr

    from oceanml3d_eval.product import open_product
    from oceanml3d_eval.regions import Region

    truth = open_product(products["truth"])
    rng = np.random.default_rng(5)
    centre = truth + rng.normal(0, 0.1, truth.u.shape).astype("f4")     # analysis, offset from truth
    members = [centre + rng.normal(0, 0.1, truth.u.shape).astype("f4") for _ in range(6)]
    ens = xr.concat(members, dim="member").assign_coords(member=np.arange(6))
    res = EnsembleScores().compute(ens, truth, Region.get("GulfStream"), "2019-01-01", "2019-01-12")
    assert res.scores["n_members"] == 6
    assert 0 < res.scores["crps_u"] < 0.1
    assert 0.5 < res.scores["spread_skill_u"] < 1.6
    assert "rank_histogram_u" in res.diagnostics


def test_deterministic_product_is_scored_as_one_member(products):
    from oceanml3d_eval.product import open_product
    from oceanml3d_eval.regions import Region

    res = EnsembleScores().compute(open_product(products["noisy"]), open_product(products["truth"]),
                                   Region.get("GulfStream"), "2019-01-01", "2019-01-12")
    assert res.scores["n_members"] == 1 and np.isfinite(res.scores["crps_u"])
    assert res.scores["spread_u"] == 0.0
