import numpy as np
import pandas as pd
import pytest

from analysis.cageatlas.service import fighter_fights, prepare_bouts
from analysis.cageatlas.similarity import FighterNotFound


def row(bout, date, me, opp, result, method, rnd, dur, **kw):
    r = dict(
        bout_id=bout, event_date=pd.Timestamp(date), weight_class="Featherweight", method=method,
        round_num=rnd, duration_sec=dur, duration_reliable=True, stats_complete=True,
        is_title_fight=False, fighter_id=me, fighter_name=me.upper(), opponent_id=opp, result=result,
        sig_landed=35.0, sig_attempted=64.0, td_landed=0.0, td_attempted=0.0, kd=1.0, ctrl_sec=0.0,
        opp_sig_landed=33.0, opp_sig_attempted=87.0, opp_td_landed=0.0, opp_td_attempted=0.0,
        opp_kd=0.0, opp_ctrl_sec=0.0,
    )
    r.update(kw)
    return r


def make_bouts():
    rows = [
        row("k1", "2024-01-01", "a", "b", "win", "KO/TKO", 2, 400.0),
        row("k1", "2024-01-01", "b", "a", "loss", "KO/TKO", 2, 400.0),
        row("d1", "2023-01-01", "a", "c", "loss", "Decision - Unanimous", 3, 900.0,
            td_landed=2.0, td_attempted=4.0, opp_td_attempted=1.0, ctrl_sec=130.0, opp_ctrl_sec=15.0, kd=0.0),
        row("d1", "2023-01-01", "c", "a", "win", "Decision - Unanimous", 3, 900.0),
        row("n1", "2022-01-01", "a", "d", "draw_nc_or_unknown", "Overturned", 1, 120.0,
            stats_complete=False, sig_landed=np.nan, is_title_fight=np.nan),
        row("n1", "2022-01-01", "d", "a", "draw_nc_or_unknown", "Overturned", 1, 120.0, stats_complete=False),
    ]
    return pd.DataFrame(rows)


def test_newest_first_and_opponent_names():
    out = fighter_fights(prepare_bouts(make_bouts()), "a")
    assert [f["date"] for f in out["fights"]] == ["2024-01-01", "2023-01-01", "2022-01-01"]
    assert [f["opponent_name"] for f in out["fights"]] == ["B", "C", "D"]
    assert out["fighter"]["fighter_name"] == "A" and out["total_fights"] == 3


def test_finish_caption_uses_only_the_numbers():
    f = fighter_fights(prepare_bouts(make_bouts()), "a")["fights"][0]
    assert f["how_won"].startswith("Won by KO/TKO (round 2 at 1:40).")
    assert "Significant strikes landed: 35 of 64 vs 33 of 87." in f["how_won"]
    assert "Knockdowns: 1 to 0." in f["how_won"]
    assert "Takedowns" not in f["how_won"] and "Control time" not in f["how_won"]
    assert f["finish_time"] == "1:40" and f["round"] == 2


def test_decision_loss_caption_with_takedowns_and_control():
    f = fighter_fights(prepare_bouts(make_bouts()), "a")["fights"][1]
    assert f["how_won"].startswith("Lost by Decision - Unanimous (went the distance, 3 rounds).")
    assert "Takedowns landed: 2 of 4 vs 0 of 1." in f["how_won"]
    assert "Control time: 2:10 vs 0:15." in f["how_won"]
    assert "Knockdowns" not in f["how_won"]


def test_missing_stats_are_none_and_caption_says_so():
    f = fighter_fights(prepare_bouts(make_bouts()), "a")["fights"][2]
    assert f["fighter"]["sig_landed"] is None          # missing, not zero
    assert f["stats_complete"] is False and "not available" in f["how_won"]
    assert f["is_title_fight"] is False                # unknown title flag -> False
    assert f["how_won"].startswith("Overturned (round 1 at 2:00).")


def test_limit_and_unknown_fighter():
    idx = prepare_bouts(make_bouts())
    out = fighter_fights(idx, "a", limit=1)
    assert len(out["fights"]) == 1 and out["total_fights"] == 3
    with pytest.raises(FighterNotFound):
        fighter_fights(idx, "nobody")
