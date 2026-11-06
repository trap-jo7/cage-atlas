import numpy as np
import pandas as pd

from analysis.cageatlas.bouts import (
    build_bouts,
    mmss_to_seconds,
    to_fighter_bouts,
    validate,
)


def make_raw():
    base = {
        "event_name": "Test", "weight_class": "Lightweight", "method": "KO/TKO",
        "f1_KD": 1.0, "f2_KD": 0.0,
        "f1_Sig_str_landed": 35.0, "f1_Sig_str_attempted": 64.0,
        "f2_Sig_str_landed": 33.0, "f2_Sig_str_attempted": 87.0,
        "f1_Td_landed": 0.0, "f1_Td_attempted": 0.0,
        "f2_Td_landed": 0.0, "f2_Td_attempted": 0.0,
        "f1_Ctrl": "00:00", "f2_Ctrl": "00:00",
    }
    rows = [
        # normal modern 2-round finish
        dict(base, event_date="2025-11-08", fighter_1="A", fighter_2="B",
             round_num=2, time="01:40",
             fight_url="http://x/fight-details/b1",
             fighter_1_url="http://x/fighter-details/a",
             fighter_2_url="http://x/fighter-details/b"),
        # legacy round-1 fight with a time over 5:00, control time unreadable
        dict(base, event_date="1998-10-16", fighter_1="C", fighter_2="D",
             round_num=1, time="07:40", f1_Ctrl="--", f2_Ctrl="--",
             fight_url="http://x/fight-details/b2",
             fighter_1_url="http://x/fighter-details/c",
             fighter_2_url="http://x/fighter-details/d"),
        # missing stats entirely, and a bad time in round 2
        dict(base, event_date="2010-01-01", fighter_1="E", fighter_2="F",
             round_num=2, time="06:10",
             f1_Sig_str_landed=np.nan, f1_Ctrl=np.nan,
             fight_url="http://x/fight-details/b3",
             fighter_1_url="http://x/fighter-details/e",
             fighter_2_url="http://x/fighter-details/f"),
    ]
    return pd.DataFrame(rows)


META = {
    "http://x/fight-details/b1": {"winner": "B", "is_title_fight": False},
    "http://x/fight-details/b2": {"winner": "C", "is_title_fight": True},
    "http://x/fight-details/b3": {"winner": "Nobody Matching", "is_title_fight": False},
    "http://x/fight-details/zzz": {"winner": "G", "is_title_fight": False},
}


def test_mmss_parsing_true_zero_vs_missing():
    assert mmss_to_seconds("00:00") == 0.0      # true zero
    assert mmss_to_seconds("01:40") == 100.0
    assert np.isnan(mmss_to_seconds("--"))      # unreadable -> missing, not 0
    assert np.isnan(mmss_to_seconds(np.nan))
    assert np.isnan(mmss_to_seconds("12:75"))   # impossible seconds


def test_duration_rules():
    b = build_bouts(make_raw(), META).set_index("bout_id")
    assert b.loc["b1", "duration_sec"] == 300 + 100      # round 2, 1:40
    assert b.loc["b2", "duration_sec"] == 460            # legacy round 1 > 5:00
    assert np.isnan(b.loc["b3", "duration_sec"])         # >5:00 after round 1
    assert bool(b.loc["b1", "duration_reliable"])
    assert not bool(b.loc["b2", "duration_reliable"])    # pre-2001 rules


def test_control_missing_vs_zero():
    b = build_bouts(make_raw(), META).set_index("bout_id")
    assert b.loc["b1", "f1_ctrl_sec"] == 0.0
    assert np.isnan(b.loc["b2", "f1_ctrl_sec"])


def test_winner_merge_and_sides():
    b = build_bouts(make_raw(), META).set_index("bout_id")
    assert b.loc["b1", "winner_side"] == "f2"
    assert b.loc["b2", "winner_side"] == "f1"
    assert b.loc["b3", "winner_side"] == "none"


def test_fighter_bouts_both_sides_and_opponent_stats():
    b = build_bouts(make_raw(), META)
    fb = to_fighter_bouts(b)
    assert len(fb) == 2 * len(b)
    row = fb[(fb.bout_id == "b1") & (fb.fighter_id == "b")].iloc[0]
    assert row["sig_landed"] == 33.0 and row["opp_sig_landed"] == 35.0
    assert row["result"] == "win"
    loser = fb[(fb.bout_id == "b1") & (fb.fighter_id == "a")].iloc[0]
    assert loser["result"] == "loss"


def test_missing_stats_not_filled_with_zero():
    b = build_bouts(make_raw(), META).set_index("bout_id")
    assert not bool(b.loc["b3", "stats_complete"])
    fb = to_fighter_bouts(b.reset_index())
    row = fb[(fb.bout_id == "b3") & (fb.fighter_id == "e")].iloc[0]
    assert np.isnan(row["sig_landed"])


def test_validation_report_counts():
    b = build_bouts(make_raw(), META)
    r = validate(b, META.keys())
    assert r["rows"] == 3
    assert r["duplicate_bout_ids"] == 0
    assert r["bouts_with_incomplete_stats"] == 1
    assert r["invalid_duration_bouts"] == 1
    assert r["json_entries_without_csv_row"] == 1
    assert r["winner_name_matches_neither_fighter"] == 1