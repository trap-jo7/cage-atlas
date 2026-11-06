import csv
import io

import numpy as np
import pytest

from analysis.cageatlas.profiles import FEATURES
from analysis.cageatlas.service import (
    _csv_safe,
    compare,
    compare_csv,
    dataset_status,
    fighter_profile,
    style_map,
)
from analysis.cageatlas.similarity import FighterNotFound
from tests.test_similarity import make_atlas


def atlas_with_map():
    a = make_atlas()
    rng = np.random.default_rng(5)
    a.artifact["pca_xy"] = rng.normal(size=(len(a.artifact["cohort_ids"]), 2))
    a.artifact["metadata"]["diagnostics"] = {"pca_explained_variance_ratio": [0.3, 0.2]}
    a.artifact["metadata"]["k"] = 3
    return a


def test_status_labels_real_snapshot_and_date():
    s = dataset_status(atlas_with_map())
    assert s["data_mode"] == "real_snapshot" and "not a live feed" in s["data_mode_label"]
    assert s["organizations"][0]["name"] == "UFC"
    assert s["model"]["groups"] == 3 and s["attribution"] and s["limits"]


def test_style_map_points_and_variance():
    a = atlas_with_map()
    m = style_map(a)
    assert len(m["points"]) == len(a.artifact["cohort_ids"])
    p0 = m["points"][0]
    assert p0["fighter_id"] == a.artifact["cohort_ids"][0]
    assert abs(p0["x"] - a.artifact["pca_xy"][0, 0]) < 1e-3
    assert m["axes"]["explained_variance_total"] == 0.5
    assert sum(g["size"] for g in m["groups"]) == len(m["points"])
    assert "two axes" in m["note"]


def test_profile_values_and_undefined_flag():
    a = atlas_with_map()
    p = fighter_profile(a, "f000")
    by = {f["feature"]: f for f in p["features"]}
    assert by["td_accuracy"]["undefined"] and by["td_accuracy"]["value"] is None
    assert not by["sig_accuracy"]["undefined"]
    assert by["sig_accuracy"]["numerator"] and by["sig_accuracy"]["unit"]
    assert p["fights"] == 5
    with pytest.raises(FighterNotFound):
        fighter_profile(a, "nope")


def test_compare_shape_percentiles_and_directions():
    a = atlas_with_map()
    r = compare(a, ["f001", "f002", "f000"])
    assert [f["fighter_id"] for f in r["fighters"]] == ["f001", "f002", "f000"]
    assert [m["feature"] for m in r["metrics"]] == FEATURES
    for m in r["metrics"]:
        assert m["higher_means"] and m["role"] in ("offensive", "defensive")
        assert len(m["values"]) == 3
        for v in m["values"]:
            assert v["percentile"] is None or 0 <= v["percentile"] <= 100
    td = next(m for m in r["metrics"] if m["feature"] == "td_accuracy")
    assert td["values"][2]["value"] is None and td["values"][2]["percentile"] is None
    assert r["reference_cohort"]["name"] and r["reference_cohort"]["size"] > 0


def test_compare_percentile_orders_with_values():
    a = atlas_with_map()
    r = compare(a, ["f001", "f002"])
    for m in r["metrics"]:
        (v1, v2) = m["values"]
        if v1["value"] is not None and v2["value"] is not None and v1["value"] != v2["value"]:
            assert (v1["value"] < v2["value"]) == (v1["percentile"] <= v2["percentile"])


def test_compare_validation():
    a = atlas_with_map()
    for bad in (["f001"], ["f001", "f001"], ["f001", "f002", "f003", "f004", "f005"], []):
        with pytest.raises(ValueError):
            compare(a, bad)
    with pytest.raises(FighterNotFound):
        compare(a, ["f001", "missing"])


def test_compare_csv_round_trip_and_safety():
    a = atlas_with_map()
    r = compare(a, ["f001", "f002", "f000"])
    rows = list(csv.reader(io.StringIO(compare_csv(r))))
    assert rows[0][:3] == ["metric", "unit", "role"] and len(rows[0]) == 6
    assert rows[1][0] == "Eligible fights" and rows[1][3:] == ["5", "5", "5"]
    td = next(row for row in rows if row[0] == "Takedown accuracy")
    assert td[5] == ""                                   # undefined -> empty, not 0
    assert _csv_safe("=SUM(A1)") == "'=SUM(A1)" and _csv_safe("Plain Name") == "Plain Name"
    assert _csv_safe(None) == ""
