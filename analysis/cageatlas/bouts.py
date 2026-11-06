"""Build clean UFC bout tables from the raw Kaggle snapshot.

Run from the project root:
    python -m analysis.cageatlas.bouts
"""
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/ufc-stats-complete-dataset")
OUT_DIR = Path("data/processed")

ROUND_SECONDS = 300  # standard 5-minute round
MODERN_RULES_START = pd.Timestamp("2001-01-01")  # unified-rules era (see docs)

# short name -> column suffix in the raw file
STATS = {
    "sig_landed": "Sig_str_landed",
    "sig_attempted": "Sig_str_attempted",
    "td_landed": "Td_landed",
    "td_attempted": "Td_attempted",
    "kd": "KD",
}
COUNT_COLS = [f"{side}_{col}" for side in ("f1", "f2") for col in STATS.values()]


def mmss_to_seconds(value):
    """'01:40' -> 100.0. Anything unreadable ('--', blank, NaN) -> NaN, never 0."""
    try:
        minutes, seconds = str(value).strip().split(":")
        minutes, seconds = int(minutes), int(seconds)
    except (ValueError, AttributeError):
        return np.nan
    if minutes < 0 or not 0 <= seconds < 60:
        return np.nan
    return float(minutes * 60 + seconds)


def load_raw(raw_dir=RAW_DIR):
    raw = pd.read_csv(raw_dir / "clean_ufc_dataset.csv")
    with open(raw_dir / "win_and_title_data-ufc.json") as f:
        meta = json.load(f)
    return raw, meta


def _last_segment(series):
    return series.str.rsplit("/", n=1).str[-1]


def build_bouts(raw, meta):
    """One row per bout, cleaned, with duration, control seconds, and winner."""
    df = raw.copy()
    df["bout_id"] = _last_segment(df["fight_url"])
    df["fighter_1_id"] = _last_segment(df["fighter_1_url"])
    df["fighter_2_id"] = _last_segment(df["fighter_2_url"])
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Duration: earlier rounds count as full 5-minute rounds. A time over 5:00
    # is only believable in round 1 (untimed-round early UFC); otherwise invalid.
    time_sec = df["time"].map(mmss_to_seconds)
    rnd = pd.to_numeric(df["round_num"], errors="coerce")
    valid = (
        time_sec.notna()
        & (time_sec > 0)
        & (rnd >= 1)
        & ((time_sec <= ROUND_SECONDS) | (rnd == 1))
    )
    df["duration_sec"] = (rnd - 1) * ROUND_SECONDS + time_sec
    df.loc[~valid, "duration_sec"] = np.nan
    df["modern_rules"] = df["event_date"] >= MODERN_RULES_START
    df["duration_reliable"] = valid & df["modern_rules"]

    # Control time: '--' and blanks are missing; '00:00' is a true zero.
    for side in ("f1", "f2"):
        df[f"{side}_ctrl_sec"] = df[f"{side}_Ctrl"].map(mmss_to_seconds)

    df["stats_complete"] = df[COUNT_COLS].notna().all(axis=1)

    # Winner and title flag come from the companion JSON, keyed by fight_url.
    meta_df = (
        pd.DataFrame.from_dict(meta, orient="index")
        .rename_axis("fight_url")
        .reset_index()
    )
    keep = ["fight_url", "winner", "is_title_fight"]
    meta_df = meta_df[[c for c in keep if c in meta_df.columns]]
    meta_df = meta_df.rename(columns={"winner": "winner_name"})
    df = df.merge(meta_df, on="fight_url", how="left")
    df["winner_side"] = np.select(
        [df["winner_name"] == df["fighter_1"], df["winner_name"] == df["fighter_2"]],
        ["f1", "f2"],
        default="none",
    )
    return df


def to_fighter_bouts(bouts):
    """One row per fighter per bout, with the opponent's numbers alongside."""
    frames = []
    for me, opp in (("f1", "f2"), ("f2", "f1")):
        n_me, n_opp = me[1], opp[1]
        out = pd.DataFrame(
            {
                "bout_id": bouts["bout_id"],
                "event_date": bouts["event_date"],
                "weight_class": bouts["weight_class"],
                "method": bouts["method"],
                "round_num": bouts["round_num"],
                "duration_sec": bouts["duration_sec"],
                "duration_reliable": bouts["duration_reliable"],
                "stats_complete": bouts["stats_complete"],
                "is_title_fight": bouts["is_title_fight"],
                "fighter_id": bouts[f"fighter_{n_me}_id"],
                "fighter_name": bouts[f"fighter_{n_me}"],
                "opponent_id": bouts[f"fighter_{n_opp}_id"],
            }
        )
        for short, col in STATS.items():
            out[short] = bouts[f"{me}_{col}"]
            out[f"opp_{short}"] = bouts[f"{opp}_{col}"]
        out["ctrl_sec"] = bouts[f"{me}_ctrl_sec"]
        out["opp_ctrl_sec"] = bouts[f"{opp}_ctrl_sec"]
        out["result"] = np.select(
            [bouts["winner_side"] == me, bouts["winner_side"] == opp],
            ["win", "loss"],
            default="draw_nc_or_unknown",
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def validate(bouts, meta_keys):
    """Counts of problems found. Nothing here silently fixes data."""
    return {
        "rows": len(bouts),
        "date_min": str(bouts["event_date"].min().date()),
        "date_max": str(bouts["event_date"].max().date()),
        "duplicate_bout_ids": int(bouts["bout_id"].duplicated().sum()),
        "missing_fighter_ids": int(
            bouts[["fighter_1_id", "fighter_2_id"]].isna().any(axis=1).sum()
        ),
        "same_fighter_both_sides": int(
            (bouts["fighter_1_id"] == bouts["fighter_2_id"]).sum()
        ),
        "bouts_with_incomplete_stats": int((~bouts["stats_complete"]).sum()),
        "sig_landed_exceeds_attempted": int(
            sum(
                (bouts[f"{s}_Sig_str_landed"] > bouts[f"{s}_Sig_str_attempted"]).sum()
                for s in ("f1", "f2")
            )
        ),
        "td_landed_exceeds_attempted": int(
            sum(
                (bouts[f"{s}_Td_landed"] > bouts[f"{s}_Td_attempted"]).sum()
                for s in ("f1", "f2")
            )
        ),
        "negative_counts": int((bouts[COUNT_COLS] < 0).sum().sum()),
        "invalid_duration_bouts": int(bouts["duration_sec"].isna().sum()),
        "legacy_rules_bouts_before_2001": int((~bouts["modern_rules"]).sum()),
        "control_time_missing_fighter_sides": int(
            bouts[["f1_ctrl_sec", "f2_ctrl_sec"]].isna().sum().sum()
        ),
        "bouts_without_winner_in_json": int(bouts["winner_name"].isna().sum()),
        "winner_name_matches_neither_fighter": int(
            (bouts["winner_name"].notna() & (bouts["winner_side"] == "none")).sum()
        ),
        "json_entries_without_csv_row": len(set(meta_keys) - set(bouts["fight_url"])),
        "fighter_1_is_winner_share": round(
            float((bouts["winner_side"] == "f1").mean()), 4
        ),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, meta = load_raw()
    bouts = build_bouts(raw, meta)
    report = validate(bouts, meta.keys())
    report["built_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["source"] = "Kaggle leandroiber/ufc-stats-complete-dataset (CC0-1.0); data from UFCStats.com"

    bouts.to_parquet(OUT_DIR / "ufc_bouts.parquet", index=False)
    to_fighter_bouts(bouts).to_parquet(OUT_DIR / "ufc_fighter_bouts.parquet", index=False)
    with open(OUT_DIR / "ufc_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
