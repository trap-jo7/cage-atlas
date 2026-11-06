import numpy as np
import pandas as pd

from analysis.cageatlas.profiles import (
    MIN_FIGHTS,
    build_profiles,
    safe_div,
)


def fight(fid, date, minutes, **kw):
    row = {
        "bout_id": f"{fid}-{date}", "event_date": pd.Timestamp(date), "fighter_id": fid,
        "fighter_name": fid.upper(), "weight_class": "Lightweight",
        "duration_sec": minutes * 60.0, "duration_reliable": True, "stats_complete": True,
        "sig_landed": 0.0, "sig_attempted": 0.0, "opp_sig_landed": 0.0, "opp_sig_attempted": 0.0,
        "td_landed": 0.0, "td_attempted": 0.0, "opp_td_landed": 0.0, "opp_td_attempted": 0.0,
        "kd": 0.0, "ctrl_sec": 0.0,
    }
    row.update(kw)
    return row


def test_safe_div_zero_denominator_is_nan_not_zero_or_inf():
    out = safe_div(pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0]))
    assert out.isna().all()


def test_rates_use_summed_counts_not_averaged_percentages():
    # Fight 1: 1/1 accurate (100%).  Fight 2: 10/100 accurate (10%).
    # Averaging percentages gives 55%; correct pooled accuracy is 11/101.
    fb = pd.DataFrame([
        fight("a", "2020-01-01", 5, sig_landed=1.0, sig_attempted=1.0),
        fight("a", "2021-01-01", 5, sig_landed=10.0, sig_attempted=100.0),
    ])
    p = build_profiles(fb).iloc[0]
    assert abs(p["sig_accuracy"] - 11 / 101) < 1e-9
    assert abs(p["sig_landed_per_min"] - 11 / 10) < 1e-9


def test_true_zero_vs_undefined():
    fb = pd.DataFrame([fight("a", "2020-01-01", 5, td_attempted=0.0, kd=0.0)])
    p = build_profiles(fb).iloc[0]
    assert p["kd_per15"] == 0.0                 # true zero knockdowns
    assert np.isnan(p["td_accuracy"])           # no attempts -> undefined
    assert np.isnan(p["td_defense"])            # never shot on -> undefined


def test_defense_uses_opponent_counts():
    fb = pd.DataFrame([
        fight("a", "2020-01-01", 5, opp_sig_landed=20.0, opp_sig_attempted=80.0),
    ])
    p = build_profiles(fb).iloc[0]
    assert abs(p["sig_defense"] - 0.75) < 1e-9


def test_ineligible_bouts_excluded():
    fb = pd.DataFrame([
        fight("a", "2020-01-01", 5, sig_landed=5.0, sig_attempted=10.0),
        fight("a", "2021-01-01", 5, sig_landed=99.0, sig_attempted=99.0,
              duration_reliable=False),
        fight("a", "2022-01-01", 5, sig_landed=99.0, sig_attempted=99.0,
              stats_complete=False),
    ])
    p = build_profiles(fb).iloc[0]
    assert p["fights"] == 1 and p["sig_landed"] == 5.0


def test_threshold_and_last5_window():
    dates = pd.date_range("2015-01-01", periods=7, freq="365D")
    fb = pd.DataFrame([fight("a", str(d.date()), 5, sig_landed=float(i),
                             sig_attempted=10.0) for i, d in enumerate(dates)])
    fb = pd.concat([fb, pd.DataFrame([fight("b", "2020-01-01", 2)])])
    career = build_profiles(fb, "career").set_index("fighter_id")
    last5 = build_profiles(fb, "last5").set_index("fighter_id")
    assert career.loc["a", "fights"] == 7 and last5.loc["a", "fights"] == 5
    assert last5.loc["a", "sig_landed"] == 2 + 3 + 4 + 5 + 6   # the latest five
    assert bool(career.loc["a", "meets_threshold"])
    assert not bool(career.loc["b", "meets_threshold"])        # 1 fight, 2 minutes
    assert MIN_FIGHTS == 3


def test_control_share_ignores_bouts_without_recorded_control():
    fb = pd.DataFrame([
        fight("a", "2020-01-01", 5, ctrl_sec=150.0),
        fight("a", "2021-01-01", 5, ctrl_sec=np.nan),   # unrecorded, not zero
    ])
    p = build_profiles(fb).iloc[0]
    assert abs(p["ctrl_share"] - 0.5) < 1e-9   # 150 s of the 300 s that had a record


def test_main_division_vs_most_recent_and_women_flag():
    fb = pd.DataFrame([
        fight("a", "2018-01-01", 5, weight_class="Featherweight"),
        fight("a", "2019-01-01", 5, weight_class="Featherweight"),
        fight("a", "2020-01-01", 5, weight_class="Lightweight"),   # most recent
        fight("w", "2018-01-01", 5, weight_class="Women's Strawweight"),
        fight("t", "2018-01-01", 5, weight_class="Lightweight"),
        fight("t", "2019-01-01", 5, weight_class="Welterweight"),    # tie -> most recent
    ])
    p = build_profiles(fb).set_index("fighter_id")
    assert p.loc["a", "weight_class"] == "Lightweight"
    assert p.loc["a", "primary_weight_class"] == "Featherweight"
    assert p.loc["t", "primary_weight_class"] == "Welterweight"
    assert bool(p.loc["w", "women_division_fighter"]) and not bool(p.loc["a", "women_division_fighter"])

