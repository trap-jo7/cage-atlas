import numpy as np
import pandas as pd

from analysis.cageatlas.model import (
    GROUP_SHARES,
    describe_clusters,
    feature_weights,
    fit_space,
    kmeans_fit,
    median_pairwise_distance,
    neighbor_overlap,
    standardize,
)
from analysis.cageatlas.profiles import FEATURE_INFO, FEATURES


def test_group_weights_sum_to_shares():
    w = dict(zip(FEATURES, feature_weights()))
    for group, share in GROUP_SHARES.items():
        total = sum(w[f] ** 2 for f in FEATURES if FEATURE_INFO[f]["group"] == group)
        assert abs(total - share) < 1e-9
    assert abs(sum(v ** 2 for v in w.values()) - 1.0) < 1e-9


def test_unweighted_is_all_ones():
    assert (feature_weights(shares=None) == 1).all()


def test_missing_values_are_median_imputed_not_zero():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, np.nan], [4.0, 40.0]])
    imputer, scaler = fit_space(X)
    filled = imputer.transform(X)
    assert filled[2, 1] == 20.0          # median of 10, 20, 40
    Z = standardize(imputer, scaler, X)
    assert abs(Z.mean(axis=0)).max() < 1e-9


def test_same_seed_gives_same_clusters():
    rng = np.random.default_rng(1)
    Z = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(5, 1, (60, 3))])
    assert (kmeans_fit(Z, 2, seed=7).labels_ == kmeans_fit(Z, 2, seed=7).labels_).all()


def test_median_pairwise_distance():
    Z = np.array([[0.0], [1.0], [3.0]])        # distances 1, 3, 2 -> median 2
    assert median_pairwise_distance(Z) == 2.0


def test_neighbor_overlap_identical_spaces_is_one():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(80, 4))
    assert neighbor_overlap(Z, Z, n_probe=40, top=5) == 1.0


def test_cluster_labels_only_when_numbers_support_them():
    z = pd.DataFrame(0.0, index=[0, 1], columns=FEATURES)
    z.loc[0, "sig_landed_per_min"] = 1.2        # clearly high
    z.loc[0, "ctrl_share"] = -0.8               # clearly low
    z.loc[1, "kd_per15"] = 0.2                  # too small to describe
    names = describe_clusters(z)
    assert "high strike volume" in names[0] and "little control time" in names[0]
    assert names[1] == "Group B: no distinct trait"
    assert "elite" not in names[0].lower() and "best" not in names[0].lower()
