import numpy as np
import pandas as pd
import pytest

from analysis.cageatlas.model import (
    describe_clusters,
    feature_weights,
    fit_space,
    kmeans_fit,
    standardize,
)
from analysis.cageatlas.profiles import FEATURES
from analysis.cageatlas.similarity import (
    FighterNotFound,
    build_atlas,
    find_similar,
    search_fighters,
)


def make_profiles(n=90, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.gamma(3.0, 1.0, (n, len(FEATURES))), columns=FEATURES)
    df["fighter_id"] = [f"f{i:03d}" for i in range(n)]
    df["fighter_name"] = [f"Fighter {i:03d}" for i in range(n)]
    df["weight_class"] = np.where(np.arange(n) % 3 == 0, "Lightweight", "Welterweight")
    df["primary_weight_class"] = df["weight_class"]
    df.loc[10, "weight_class"] = "Featherweight"     # recent division differs from main one
    df["women_division_fighter"] = (np.arange(n) % 4 == 0)
    df["fights"] = 5
    df["minutes"] = np.linspace(20.0, 120.0, n)
    df["first_fight"] = pd.Timestamp("2015-01-01")
    df["last_fight"] = pd.Timestamp("2024-01-01")
    df["meets_threshold"] = True
    df.loc[n - 1, "meets_threshold"] = False        # one low-evidence fighter
    df.loc[0, "td_accuracy"] = np.nan                # one undefined value
    return df


def make_atlas(profiles=None, k=3):
    p = make_profiles() if profiles is None else profiles
    cohort = p[p["meets_threshold"]].reset_index(drop=True)
    X = cohort[FEATURES].to_numpy(dtype=float)
    imputer, scaler = fit_space(X)
    Zstd = standardize(imputer, scaler, X)
    weights = feature_weights()
    km = kmeans_fit(Zstd * weights, k)
    cz = pd.DataFrame(Zstd, columns=FEATURES).groupby(km.labels_).mean()
    artifact = {
        "features": FEATURES, "weights": weights, "imputer": imputer, "scaler": scaler,
        "kmeans": km, "cohort_ids": cohort["fighter_id"].tolist(), "labels": km.labels_,
        "cluster_names": describe_clusters(cz),
        "pca_xy": np.zeros((len(cohort), 2)), "pca": None, "d_ref": 1.5,
        "metadata": {"cohort": {"definition": "test cohort", "size": len(cohort)},
                     "source_snapshot": {"date_max": "2026-05-16"}},
    }
    return build_atlas(artifact, p)


def test_selected_fighter_is_excluded():
    a = make_atlas()
    res = find_similar(a, "f010", n=100)
    assert "f010" not in [r["fighter_id"] for r in res["results"]]


def test_results_are_ordered_and_match_brute_force():
    a = make_atlas()
    res = find_similar(a, "f010", n=15)
    d = [r["distance"] for r in res["results"]]
    assert d == sorted(d)
    q = a.row_of["f010"]
    brute = np.linalg.norm(a.Z - a.Z[q], axis=1)
    brute[q] = np.inf
    brute[~a.in_cohort] = np.inf
    expected = [a.profiles.at[i, "fighter_id"] for i in np.argsort(brute, kind="stable")[:15]]
    assert [r["fighter_id"] for r in res["results"]] == expected


def test_deterministic():
    a = make_atlas()
    assert find_similar(a, "f010") == find_similar(a, "f010")
    b = make_atlas()
    assert [r["fighter_id"] for r in find_similar(a, "f010")["results"]] == \
           [r["fighter_id"] for r in find_similar(b, "f010")["results"]]


def test_never_uses_map_coordinates():
    a = make_atlas()
    before = find_similar(a, "f010")
    a.artifact["pca_xy"] = np.random.default_rng(9).normal(size=(len(a.artifact["pca_xy"]), 2))
    a.artifact["pca"] = "garbage"
    assert find_similar(a, "f010") == before


def test_filters_same_cluster_and_division():
    a = make_atlas()
    res = find_similar(a, "f010", n=100, same_cluster=True)
    assert res["results"] and all(r["same_group"] for r in res["results"])
    res = find_similar(a, "f010", n=100, division="same")
    own = a.profiles.at[a.row_of["f010"], "primary_weight_class"]
    assert res["results"] and all(r["weight_class"] == own for r in res["results"])


def test_index_calibration_does_not_change_with_filters():
    a = make_atlas()
    open_ = {r["fighter_id"]: r for r in find_similar(a, "f010", n=100)["results"]}
    filt = find_similar(a, "f010", n=100, division="same")["results"]
    assert filt
    for r in filt:
        assert r["similarity_index"] == open_[r["fighter_id"]]["similarity_index"]
        assert r["distance"] == open_[r["fighter_id"]]["distance"]


def test_empty_candidate_set_gives_reason():
    a = make_atlas()
    res = find_similar(a, "f010", division="Strawweight")
    assert res["results"] == [] and res["empty_reason"]


def test_invalid_inputs():
    a = make_atlas()
    with pytest.raises(FighterNotFound):
        find_similar(a, "nope")
    for bad in (0, 101, -3, "5", 2.5, True):
        with pytest.raises(ValueError):
            find_similar(a, "f010", n=bad)
    with pytest.raises(ValueError):
        find_similar(a, "f010", division=5)


def test_low_evidence_fighter_warns_and_candidates_stay_in_cohort():
    a = make_atlas()
    last = a.profiles.at[len(a.profiles) - 1, "fighter_id"]
    res = find_similar(a, last, n=100)
    assert res["query"]["evidence_warning"] and not res["query"]["in_model_cohort"]
    assert last not in [r["fighter_id"] for r in res["results"]]
    assert len(res["results"]) == res["comparison_cohort"]["size"]


def test_explanations_come_from_differences_and_skip_undefined():
    a = make_atlas()
    res = find_similar(a, "f000", n=100)                    # f000 has an undefined value
    for r in res["results"]:
        used = [i["feature"] for i in r["closest_characteristics"] + r["largest_differences"]]
        assert "td_accuracy" not in used
        assert len(r["closest_characteristics"]) == 3 and len(r["largest_differences"]) == 2
        closest = [abs(i["std_difference"]) for i in r["closest_characteristics"]]
        largest = [abs(i["std_difference"]) for i in r["largest_differences"]]
        assert max(closest) <= min(largest) + 0.01


def test_search_by_name():
    a = make_atlas()
    hits = search_fighters(a, "fighter 01")
    assert hits and all("fighter 01" in h["fighter_name"].lower() for h in hits)
    assert search_fighters(a, "x") == [] and search_fighters(a, None) == []


def test_division_uses_main_division_not_most_recent():
    a = make_atlas()
    res = find_similar(a, "f010", n=100, division="same")      # f010: main = Welterweight
    assert res["query"]["weight_class"] == "Welterweight"
    assert res["query"]["most_recent_weight_class"] == "Featherweight"
    assert res["results"] and all(r["weight_class"] == "Welterweight" for r in res["results"])


def test_category_filter():
    a = make_atlas()
    women = set(a.profiles.loc[a.profiles["women_division_fighter"], "fighter_id"])
    res = find_similar(a, "f010", n=100, category="women")
    assert res["results"] and all(r["fighter_id"] in women for r in res["results"])
    res = find_similar(a, "f010", n=100, category="men")
    assert res["results"] and all(r["fighter_id"] not in women for r in res["results"])
    assert find_similar(a, "f010", category="same")["comparison_cohort"]["filters"]["category"] == "men"
    with pytest.raises(ValueError):
        find_similar(a, "f010", category="everyone")


def test_min_minutes_filter():
    a = make_atlas()
    res = find_similar(a, "f010", n=100, min_minutes=80)
    assert res["results"] and all(r["minutes"] >= 80 for r in res["results"])
    for bad in (-1, "10", True):
        with pytest.raises(ValueError):
            find_similar(a, "f010", min_minutes=bad)
