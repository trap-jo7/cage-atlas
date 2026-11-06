from analysis.cageatlas.service import prepare_bouts, recent_activity
from tests.test_fights import row
from tests.test_similarity import make_atlas
import pandas as pd


def make_index():
    rows = [
        row("r1", "2024-05-01", "f001", "f002", "win", "KO/TKO", 1, 90.0),
        row("r1", "2024-05-01", "f002", "f001", "loss", "KO/TKO", 1, 90.0),
        row("r2", "2024-05-01", "f003", "f004", "draw_nc_or_unknown", "Decision - Majority", 3, 900.0),
        row("r2", "2024-05-01", "f004", "f003", "draw_nc_or_unknown", "Decision - Majority", 3, 900.0),
        row("r3", "2024-04-01", "f005", "f006", "win", "Submission", 2, 450.0, is_title_fight=True),
        row("r3", "2024-04-01", "f006", "f005", "loss", "Submission", 2, 450.0, is_title_fight=True),
        # the one low-evidence fighter in the test atlas fought most recently of all but one
        row("r4", "2024-04-15", "f089", "f001", "win", "KO/TKO", 1, 60.0),
    ]
    return prepare_bouts(pd.DataFrame(rows))


def test_results_come_only_from_the_latest_event_and_name_the_winner_first():
    out = recent_activity(make_atlas(), make_index())
    assert out["latest_event_date"] == "2024-05-01"
    by = {r["bout_id"]: r for r in out["results"]}
    assert set(by) == {"r1", "r2"}
    assert by["r1"]["outcome"] == "a_won" and by["r1"]["a_id"] == "f001" and by["r1"]["b_id"] == "f002"
    assert by["r1"]["finish_time"] == "1:30" and by["r1"]["round"] == 1
    assert by["r2"]["outcome"] == "draw_nc_or_unknown"


def test_recent_fighters_are_cohort_only_ordered_and_capped_per_event():
    a = make_atlas()
    out = recent_activity(a, make_index(), per_event=2)
    ids = [f["fighter_id"] for f in out["fighters"]]
    assert "f089" not in ids                                   # low-evidence fighter excluded
    dates = [f["last_fight"] for f in out["fighters"]]
    assert dates == sorted(dates, reverse=True)
    assert dates.count("2024-05-01") <= 2
    assert out["note"] and "not a popularity" in out["note"]
    assert all(f["group"].startswith("Group") for f in out["fighters"])


def test_data_through_comes_from_the_model_snapshot():
    assert recent_activity(make_atlas(), make_index())["data_through"] == "2026-05-16"
