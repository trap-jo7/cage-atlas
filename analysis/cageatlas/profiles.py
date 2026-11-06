"""Fighter style profiles: add up the counts first, then compute rates.

Run from the project root (after analysis.cageatlas.bouts):
    python -m analysis.cageatlas.profiles
"""
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROC_DIR = Path("data/processed")
MIN_FIGHTS = 3
MIN_MINUTES = 15.0

# Every feature: what it is, its unit, and the exact numerator / denominator.
FEATURE_INFO = {
    "sig_landed_per_min": {"group": "striking", "unit": "per minute",
        "num": "significant strikes landed", "den": "fight minutes"},
    "sig_absorbed_per_min": {"group": "striking", "unit": "per minute",
        "num": "significant strikes absorbed (opponent landed)", "den": "fight minutes"},
    "sig_accuracy": {"group": "striking", "unit": "share (0-1)",
        "num": "significant strikes landed", "den": "significant strikes attempted"},
    "sig_defense": {"group": "striking", "unit": "share (0-1)",
        "num": "1 - opponent sig. strikes landed", "den": "opponent sig. strikes attempted"},
    "kd_per15": {"group": "striking", "unit": "per 15 minutes",
        "num": "knockdowns scored", "den": "fight minutes / 15"},
    "td_attempted_per15": {"group": "grappling", "unit": "per 15 minutes",
        "num": "takedowns attempted", "den": "fight minutes / 15"},
    "td_accuracy": {"group": "grappling", "unit": "share (0-1)",
        "num": "takedowns landed", "den": "takedowns attempted"},
    "td_defense": {"group": "grappling", "unit": "share (0-1)",
        "num": "1 - opponent takedowns landed", "den": "opponent takedowns attempted"},
    "ctrl_share": {"group": "control", "unit": "share (0-1)",
        "num": "control time (seconds)", "den": "fight time with recorded control (seconds)"},
}
FEATURES = list(FEATURE_INFO)

SUM_COLS = [
    "sig_landed", "sig_attempted", "opp_sig_landed", "opp_sig_attempted",
    "td_landed", "td_attempted", "opp_td_landed", "opp_td_attempted", "kd",
]


def safe_div(num, den):
    """num / den, but NaN (undefined) when the denominator is 0. Never 0, never inf."""
    return num / den.where(den != 0)


def eligible_bouts(fighter_bouts):
    """Bouts that can support a style profile: complete stats, trustworthy duration."""
    return fighter_bouts[
        fighter_bouts["stats_complete"] & fighter_bouts["duration_reliable"]
    ].copy()


def select_window(rows, window):
    rows = rows.sort_values(["fighter_id", "event_date", "bout_id"])
    if window == "career":
        return rows
    if window == "last5":
        return rows.groupby("fighter_id").tail(5)
    raise ValueError(f"unknown window: {window}")


def primary_division(divisions):
    """Division with the most fights; ties go to the most recent one."""
    s = divisions.dropna()
    if s.empty:
        return None
    counts = s.value_counts()
    tied = set(counts[counts == counts.max()].index)
    for value in reversed(s.tolist()):
        if value in tied:
            return value


def aggregate(rows):
    rows = rows.sort_values(["fighter_id", "event_date", "bout_id"]).copy()
    # Control share only counts bouts where control time was actually recorded.
    known = rows["ctrl_sec"].notna()
    rows["ctrl_num_sec"] = rows["ctrl_sec"].fillna(0)
    rows["ctrl_den_sec"] = rows["duration_sec"].where(known, 0)

    g = rows.groupby("fighter_id")
    out = g[SUM_COLS + ["ctrl_num_sec", "ctrl_den_sec"]].sum()
    out["fights"] = g.size()
    out["minutes"] = g["duration_sec"].sum() / 60
    out["first_fight"] = g["event_date"].min()
    out["last_fight"] = g["event_date"].max()
    out["fighter_name"] = g["fighter_name"].last()
    out["weight_class"] = g["weight_class"].last()  # most recent division
    out["primary_weight_class"] = g["weight_class"].agg(primary_division)
    out["women_division_fighter"] = g["weight_class"].agg(
        lambda s: bool(s.dropna().str.startswith("Women's").any())
    )
    return out.reset_index()


def add_features(p):
    m = p["minutes"]
    p["sig_landed_per_min"] = safe_div(p["sig_landed"], m)
    p["sig_absorbed_per_min"] = safe_div(p["opp_sig_landed"], m)
    p["sig_accuracy"] = safe_div(p["sig_landed"], p["sig_attempted"])
    p["sig_defense"] = 1 - safe_div(p["opp_sig_landed"], p["opp_sig_attempted"])
    p["kd_per15"] = safe_div(p["kd"], m) * 15
    p["td_attempted_per15"] = safe_div(p["td_attempted"], m) * 15
    p["td_accuracy"] = safe_div(p["td_landed"], p["td_attempted"])
    p["td_defense"] = 1 - safe_div(p["opp_td_landed"], p["opp_td_attempted"])
    p["ctrl_share"] = safe_div(p["ctrl_num_sec"], p["ctrl_den_sec"])
    p["meets_threshold"] = (p["fights"] >= MIN_FIGHTS) & (p["minutes"] >= MIN_MINUTES)
    return p


def build_profiles(fighter_bouts, window="career"):
    rows = select_window(eligible_bouts(fighter_bouts), window)
    return add_features(aggregate(rows))


def correlated_pairs(cohort, cutoff=0.7):
    corr = cohort[FEATURES].corr()
    pairs = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= cutoff:
                pairs.append({"a": a, "b": b, "r": round(float(r), 3)})
    return pairs


def coverage_report(profiles):
    cohort = profiles[profiles["meets_threshold"]]
    return {
        "profiles_total": len(profiles),
        "profiles_meeting_threshold": len(cohort),
        "threshold": {"min_fights": MIN_FIGHTS, "min_minutes": MIN_MINUTES},
        "undefined_feature_counts_in_cohort": {
            f: int(cohort[f].isna().sum()) for f in FEATURES
        },
        "feature_summary_in_cohort": {
            f: {k: round(float(v), 3) for k, v in
                cohort[f].describe()[["min", "50%", "max"]].items()}
            for f in FEATURES
        },
        "highly_correlated_pairs_abs_r_ge_0.7": correlated_pairs(cohort),
    }


def main():
    fb = pd.read_parquet(PROC_DIR / "ufc_fighter_bouts.parquet")
    report = {"built_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    for window in ("career", "last5"):
        profiles = build_profiles(fb, window)
        profiles.to_parquet(PROC_DIR / f"ufc_profiles_{window}.parquet", index=False)
        report[window] = coverage_report(profiles)
    with open(PROC_DIR / "ufc_profiles_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

