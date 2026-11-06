"""Plain-Python functions behind the website API (and reusable by Streamlit).

Everything here works on an already-loaded Atlas, so a web request never trains a
model, reads raw data, or scrapes anything.
"""
import csv
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import PROC_DIR
from .profiles import FEATURE_INFO, FEATURES
from .similarity import (
    LABELS,
    LIMITS,
    FighterNotFound,
    _category,
    _div_col,
    _group_name,
    _num,
    _query_summary,
)

ATTRIBUTION = (
    "UFC statistics originally from UFCStats.com, compiled and published on Kaggle by "
    "leandroiber under CC0 1.0. This is a snapshot, not a live feed. CAGE ATLAS is an "
    "independent analysis project, not affiliated with or endorsed by the UFC or UFCStats."
)

# Plain-language direction for every metric. "Role" says whether the number describes
# what a fighter does (offensive) or what happens to them (defensive).
METRIC_NOTES = {
    "sig_landed_per_min": ("offensive", "Higher = lands more significant strikes per minute"),
    "sig_absorbed_per_min": ("defensive", "Higher = absorbs more significant strikes per minute"),
    "sig_accuracy": ("offensive", "Higher = a larger share of attempted strikes land"),
    "sig_defense": ("defensive", "Higher = opponents land a smaller share of their strikes"),
    "kd_per15": ("offensive", "Higher = scores more knockdowns per 15 minutes"),
    "td_attempted_per15": ("offensive", "Higher = attempts more takedowns per 15 minutes"),
    "td_accuracy": ("offensive", "Higher = a larger share of takedown attempts succeed"),
    "td_defense": ("defensive", "Higher = opponents' takedown attempts succeed less often"),
    "ctrl_share": ("offensive", "Higher = spends more of the fight time in control"),
}

MAP_NOTE = (
    "The map squeezes nine measurements into two axes, so it shows only part of the "
    "picture. Two fighters close together on the map may still differ; the match "
    "list, not the map, is the real similarity ranking."
)


def dataset_status(atlas):
    meta = atlas.artifact["metadata"]
    snap = meta.get("source_snapshot") or {}
    diag = meta.get("diagnostics") or {}
    return {
        "data_mode": "real_snapshot",
        "data_mode_label": "Real data snapshot (not a live feed)",
        "organizations": [{
            "name": "UFC",
            "style_model": True,
            "data_through": snap.get("date_max"),
            "source": snap.get("source"),
        }],
        "profiles_total": int(len(atlas.profiles)),
        "model_cohort_size": (meta.get("cohort") or {}).get("size"),
        "model": {
            "groups": meta.get("k"),
            "trained_at_utc": meta.get("trained_at_utc"),
            "silhouette": diag.get("silhouette"),
            "stability_mean_ari": diag.get("stability_mean_ari"),
        },
        "attribution": ATTRIBUTION,
        "limits": LIMITS,
    }


def style_map(atlas):
    """2-D map points for the model cohort. For display only; never used for matching."""
    art = atlas.artifact
    meta = art["metadata"]
    diag = meta.get("diagnostics") or {}
    names = art["cluster_names"]
    prof = atlas.profiles.set_index("fighter_id")
    div = _div_col(atlas)
    xy = np.asarray(art["pca_xy"])
    points = []
    for i, fid in enumerate(art["cohort_ids"]):
        c = int(art["labels"][i])
        points.append({
            "fighter_id": fid,
            "fighter_name": prof.at[fid, "fighter_name"],
            "weight_class": prof.at[fid, div],
            "cluster": c,
            "group": names[c],
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
        })
    ratio = list(diag.get("pca_explained_variance_ratio") or [])
    sizes = np.bincount(np.asarray(art["labels"]), minlength=len(names))
    return {
        "points": points,
        "axes": {"explained_variance_ratio": ratio,
                 "explained_variance_total": round(float(sum(ratio)), 4) if ratio else None},
        "groups": [{"cluster": int(c), "name": n, "size": int(sizes[c])}
                   for c, n in sorted(names.items())],
        "note": MAP_NOTE,
    }


def fighter_profile(atlas, fighter_id):
    q = atlas.row_of.get(fighter_id)
    if q is None:
        raise FighterNotFound(fighter_id)
    p = atlas.profiles.iloc[q]
    out = _query_summary(atlas, q)
    out["window"] = "career"
    out["features"] = [{
        "feature": f,
        "label": LABELS[f],
        "unit": FEATURE_INFO[f]["unit"],
        "group": FEATURE_INFO[f]["group"],
        "role": METRIC_NOTES[f][0],
        "higher_means": METRIC_NOTES[f][1],
        "value": _num(p[f]),
        "undefined": bool(atlas.imputed[q, i]),
        "numerator": FEATURE_INFO[f]["num"],
        "denominator": FEATURE_INFO[f]["den"],
    } for i, f in enumerate(FEATURES)]
    return out


def compare(atlas, fighter_ids):
    """Side-by-side raw values for 2 to 4 fighters, plus percentile in the model cohort."""
    ids = list(fighter_ids)
    if not 2 <= len(ids) <= 4 or len(set(ids)) != len(ids):
        raise ValueError("choose 2 to 4 different fighters")
    rows = []
    for fid in ids:
        q = atlas.row_of.get(fid)
        if q is None:
            raise FighterNotFound(fid)
        rows.append(q)

    prof = atlas.profiles
    cohort = prof[atlas.in_cohort]
    fighters = []
    for q in rows:
        s = _query_summary(atlas, q)
        fighters.append({k: s[k] for k in (
            "fighter_id", "fighter_name", "weight_class", "category", "fights", "minutes",
            "first_fight", "last_fight", "in_model_cohort", "undefined_features",
            "evidence_warning")})

    metrics = []
    for i, f in enumerate(FEATURES):
        ref = cohort[f].dropna().to_numpy()
        values = []
        for q in rows:
            undefined = bool(atlas.imputed[q, i])
            v = None if undefined else _num(prof.iloc[q][f])
            pct = None if v is None or len(ref) == 0 else round(float((ref <= v).mean() * 100), 1)
            values.append({"fighter_id": prof.iloc[q]["fighter_id"], "value": v,
                           "percentile": pct, "undefined": undefined})
        metrics.append({
            "feature": f, "label": LABELS[f], "unit": FEATURE_INFO[f]["unit"],
            "group": FEATURE_INFO[f]["group"], "role": METRIC_NOTES[f][0],
            "higher_means": METRIC_NOTES[f][1], "values": values,
        })
    meta = atlas.artifact["metadata"]
    return {
        "fighters": fighters,
        "metrics": metrics,
        "reference_cohort": {
            "name": "UFC model cohort",
            "size": int(len(cohort)),
            "definition": meta["cohort"]["definition"],
            "percentile_meaning": "share of cohort fighters at or below this value",
        },
        "limits": LIMITS,
    }


def _csv_safe(value):
    """Stop spreadsheet programs from running a cell that starts like a formula."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def compare_csv(result):
    """The displayed comparison as CSV text: raw values, units, sample sizes."""
    buf = io.StringIO()
    w = csv.writer(buf)
    fighters = result["fighters"]
    w.writerow(["metric", "unit", "role"] + [_csv_safe(f["fighter_name"]) for f in fighters])
    w.writerow(["Eligible fights", "count", ""] + [f["fights"] for f in fighters])
    w.writerow(["Recorded fight time", "minutes", ""] + [f["minutes"] for f in fighters])
    for m in result["metrics"]:
        w.writerow([m["label"], m["unit"], m["role"]] +
                   ["" if v["value"] is None else v["value"] for v in m["values"]])
    w.writerow([])
    w.writerow(["Empty cell = undefined (no attempts), not zero."])
    w.writerow([result["reference_cohort"]["name"] + ": " +
                str(result["reference_cohort"]["size"]) + " fighters"])
    return buf.getvalue()


FIGHTS_NOTE = (
    "Totals are for the whole fight; round-by-round detail is not in this dataset. In the "
    "replay, the real counts are spread across eight seconds, so the order of events inside "
    "it is illustrative."
)


@dataclass
class BoutIndex:
    frame: pd.DataFrame
    names: dict


def prepare_bouts(frame):
    """Newest fights first, plus an id -> name map so opponents can be named."""
    f = frame.sort_values(["event_date", "bout_id"], ascending=[False, True]).reset_index(drop=True)
    return BoutIndex(f, dict(zip(f["fighter_id"], f["fighter_name"])))


def load_bouts():
    return prepare_bouts(pd.read_parquet(PROC_DIR / "ufc_fighter_bouts.parquet"))


def _clock(sec):
    if sec is None or pd.isna(sec):
        return None
    s = int(round(float(sec)))
    return f"{s // 60}:{s % 60:02d}"


def _s(x):
    return None if pd.isna(x) else str(x)


def _count(x):
    """Whole counts stay whole (35, not 35.0); missing stays None, never 0."""
    if pd.isna(x):
        return None
    return int(x) if float(x).is_integer() else round(float(x), 2)


def _fight_stats(row, prefix):
    return {k: _count(row[f"{prefix}{k}"])
            for k in ("sig_landed", "sig_attempted", "td_landed", "td_attempted", "kd", "ctrl_sec")}


def _how_won(row, me, opp):
    """A caption built only from the numbers: result, method, and the totals."""
    method = str(row["method"])
    r = int(row["round_num"])
    dur = row["duration_sec"]
    if method.startswith("Decision"):
        when = f"went the distance, {r} rounds"
    else:
        when = f"round {r}" + ("" if pd.isna(dur) else f" at {_clock(dur - (r - 1) * 300)}")
    verb = {"win": "Won", "loss": "Lost"}.get(row["result"])
    head = f"{verb} by {method} ({when})." if verb else f"{method} ({when})."
    if not bool(row["stats_complete"]):
        return head + " Strike, takedown and control totals are not available for this fight."
    parts = [f"Significant strikes landed: {me['sig_landed']} of {me['sig_attempted']} "
             f"vs {opp['sig_landed']} of {opp['sig_attempted']}."]
    if (me["kd"] or 0) or (opp["kd"] or 0):
        parts.append(f"Knockdowns: {me['kd']} to {opp['kd']}.")
    if (me["td_attempted"] or 0) or (opp["td_attempted"] or 0):
        parts.append(f"Takedowns landed: {me['td_landed']} of {me['td_attempted']} "
                     f"vs {opp['td_landed']} of {opp['td_attempted']}.")
    if (me["ctrl_sec"] or 0) >= 30 or (opp["ctrl_sec"] or 0) >= 30:
        parts.append(f"Control time: {_clock(me['ctrl_sec'] or 0)} vs {_clock(opp['ctrl_sec'] or 0)}.")
    return " ".join([head] + parts)


def fighter_fights(index, fighter_id, limit=60):
    """One fighter's past fights, newest first, with both sides' totals."""
    rows = index.frame[index.frame["fighter_id"] == fighter_id]
    if rows.empty:
        raise FighterNotFound(fighter_id)
    fights = []
    for _, r in rows.head(limit).iterrows():
        me, opp = _fight_stats(r, ""), _fight_stats(r, "opp_")
        dur = None if pd.isna(r["duration_sec"]) else float(r["duration_sec"])
        rnd = int(r["round_num"])
        title = r["is_title_fight"]
        fights.append({
            "bout_id": r["bout_id"],
            "date": str(r["event_date"].date()),
            "opponent_id": r["opponent_id"],
            "opponent_name": index.names.get(r["opponent_id"]),
            "weight_class": _s(r["weight_class"]),
            "result": r["result"],
            "method": _s(r["method"]),
            "round": rnd,
            "finish_time": None if dur is None else _clock(dur - (rnd - 1) * 300),
            "is_title_fight": bool(title) if pd.notna(title) else False,
            "duration_sec": dur,
            "stats_complete": bool(r["stats_complete"]),
            "fighter": me,
            "opponent": opp,
            "how_won": _how_won(r, me, opp),
        })
    return {
        "fighter": {"fighter_id": fighter_id, "fighter_name": _s(rows.iloc[0]["fighter_name"])},
        "total_fights": int(len(rows)),
        "fights": fights,
        "note": FIGHTS_NOTE,
    }


RECENT_NOTE = (
    "Latest results and recently active fighters come from the most recent fights in this "
    "snapshot. This is not a popularity or trending ranking; the data has no view or search counts."
)


def recent_activity(atlas, index, n_results=14, n_fighters=12, per_event=4):
    """The newest event in the data, plus recently active fighters (at most a few per event)."""
    f = index.frame
    latest = f["event_date"].max()
    day = f[f["event_date"] == latest].copy()
    day["_w"] = (day["result"] == "win").astype(int)
    day = day.sort_values(["bout_id", "_w"], ascending=[True, False]).drop_duplicates("bout_id")
    results = []
    for _, r in day.head(n_results).iterrows():
        dur = None if pd.isna(r["duration_sec"]) else float(r["duration_sec"])
        rnd = int(r["round_num"])
        title = r["is_title_fight"]
        results.append({
            "bout_id": r["bout_id"],
            "a_id": r["fighter_id"], "a_name": _s(r["fighter_name"]),
            "b_id": r["opponent_id"], "b_name": index.names.get(r["opponent_id"]),
            "outcome": "a_won" if r["result"] == "win" else "draw_nc_or_unknown",
            "method": _s(r["method"]), "round": rnd,
            "finish_time": None if dur is None else _clock(dur - (rnd - 1) * 300),
            "weight_class": _s(r["weight_class"]),
            "is_title_fight": bool(title) if pd.notna(title) else False,
        })
    last = f.groupby("fighter_id")["event_date"].max()
    cohort = atlas.profiles[atlas.in_cohort].copy()
    cohort["_last"] = cohort["fighter_id"].map(last)
    cohort = cohort.dropna(subset=["_last"]).sort_values(["_last", "minutes"], ascending=[False, False])
    cohort = cohort.groupby("_last", sort=False).head(per_event).head(n_fighters)
    div = _div_col(atlas)
    fighters = [{
        "fighter_id": r["fighter_id"], "fighter_name": _s(r["fighter_name"]),
        "weight_class": _s(r[div]), "last_fight": str(r["_last"].date()),
        "group": _group_name(atlas, atlas.row_of[r["fighter_id"]]), "fights": int(r["fights"]),
    } for _, r in cohort.iterrows()]
    snap = atlas.artifact["metadata"].get("source_snapshot") or {}
    return {"data_through": snap.get("date_max"), "latest_event_date": str(latest.date()),
            "results": results, "fighters": fighters, "note": RECENT_NOTE}
