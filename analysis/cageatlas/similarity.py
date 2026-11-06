"""Nearest-fighter search in the model's own feature space.

Distances use the weighted, standardized features the model was fitted on.
The 2-D map coordinates (PCA) are never used here.

Try it from the project root:
    python -m analysis.cageatlas.similarity "Max Holloway"
    python -m analysis.cageatlas.similarity "Max Holloway" --same-cluster --division same
"""
import argparse
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .model import MODEL_DIR, PROC_DIR, standardize
from .profiles import FEATURE_INFO, FEATURES

LABELS = {
    "sig_landed_per_min": "Significant strikes landed",
    "sig_absorbed_per_min": "Significant strikes absorbed",
    "sig_accuracy": "Striking accuracy",
    "sig_defense": "Striking defense",
    "kd_per15": "Knockdowns",
    "td_attempted_per15": "Takedown attempts",
    "td_accuracy": "Takedown accuracy",
    "td_defense": "Takedown defense",
    "ctrl_share": "Control time share",
}

LIMITS = (
    "Aggregate statistics cannot fully capture timing, footwork, tactical intent, "
    "opponent quality, or how a fighter's style has changed. Similarity describes "
    "statistical resemblance; it says nothing about who would win."
)
INDEX_NOTE = (
    "Similarity index = 100 x exp(-distance / reference distance), where the reference "
    "is the median distance between any two fighters in the model cohort. It is a "
    "fixed scale, not a probability, and it does not change with filters."
)
LOW_EVIDENCE = (
    "Low evidence: fewer than 3 eligible fights or 15 recorded minutes. Shown for "
    "reference only; this fighter was not part of the model's training cohort."
)


class FighterNotFound(KeyError):
    pass


class StaleModelError(RuntimeError):
    pass


@dataclass
class Atlas:
    artifact: dict
    profiles: pd.DataFrame   # every fighter with a profile, one row each
    Z: np.ndarray            # weighted + standardized space (used for distance)
    Zstd: np.ndarray         # standardized only (used for plain-language explanations)
    imputed: np.ndarray      # True where a value was undefined and median-filled
    in_cohort: np.ndarray    # True for fighters in the model's training cohort
    cluster: np.ndarray      # group id per fighter
    row_of: dict             # fighter_id -> row number


def build_atlas(artifact, profiles, check_consistency=True):
    profiles = profiles.reset_index(drop=True)
    X = profiles[artifact["features"]].to_numpy(dtype=float)
    Zstd = standardize(artifact["imputer"], artifact["scaler"], X)
    Z = Zstd * artifact["weights"]
    row_of = {fid: i for i, fid in enumerate(profiles["fighter_id"])}

    cohort_ids = list(artifact["cohort_ids"])
    in_cohort = profiles["fighter_id"].isin(set(cohort_ids)).to_numpy()
    cluster = artifact["kmeans"].predict(Z)
    if check_consistency:
        missing = [fid for fid in cohort_ids if fid not in row_of]
        if missing:
            raise StaleModelError("profiles no longer contain the model's cohort; retrain")
        rows = [row_of[fid] for fid in cohort_ids]
        if not (artifact["kmeans"].predict(Z[rows]) == np.asarray(artifact["labels"])).all():
            raise StaleModelError("profiles changed since the model was trained; retrain")
        cluster[rows] = artifact["labels"]
    return Atlas(artifact, profiles, Z, Zstd, np.isnan(X), in_cohort, cluster, row_of)


def load_atlas(window="career"):
    artifact = joblib.load(MODEL_DIR / "ufc_career_model.joblib")
    profiles = pd.read_parquet(PROC_DIR / f"ufc_profiles_{window}.parquet")
    return build_atlas(artifact, profiles, check_consistency=(window == "career"))


def _num(x):
    return None if pd.isna(x) else round(float(x), 3)


def _group_name(atlas, row):
    return atlas.artifact["cluster_names"][int(atlas.cluster[row])]


def _imputed_features(atlas, row):
    return [f for i, f in enumerate(FEATURES) if atlas.imputed[row, i]]


def _div_col(atlas):
    """Divisions are matched on the division a fighter fought in most (falls back to
    the most recent one for older profile files)."""
    return "primary_weight_class" if "primary_weight_class" in atlas.profiles else "weight_class"


def _category(atlas, row):
    if "women_division_fighter" not in atlas.profiles:
        return None
    return "women" if bool(atlas.profiles.iloc[row]["women_division_fighter"]) else "men"


def _query_summary(atlas, q):
    p = atlas.profiles.iloc[q]
    return {
        "fighter_id": p["fighter_id"],
        "fighter_name": p["fighter_name"],
        "weight_class": p[_div_col(atlas)],
        "most_recent_weight_class": p["weight_class"],
        "category": _category(atlas, q),
        "fights": int(p["fights"]),
        "minutes": round(float(p["minutes"]), 1),
        "first_fight": str(p["first_fight"].date()),
        "last_fight": str(p["last_fight"].date()),
        "group": _group_name(atlas, q),
        "in_model_cohort": bool(atlas.in_cohort[q]),
        "undefined_features": _imputed_features(atlas, q),
        "evidence_warning": None if atlas.in_cohort[q] else LOW_EVIDENCE,
    }


def _item(atlas, q, c, i):
    f = FEATURES[i]
    diff = float(atlas.Zstd[c, i] - atlas.Zstd[q, i])
    prof = atlas.profiles
    return {
        "feature": f,
        "label": LABELS[f],
        "unit": FEATURE_INFO[f]["unit"],
        "query_value": _num(prof.iloc[q][f]),
        "match_value": _num(prof.iloc[c][f]),
        "std_difference": round(diff, 2),  # match minus query, in cohort std devs
    }


def _explain(atlas, q, c):
    """3 closest and 2 most different features, straight from the standardized gaps.
    Features where either fighter's value was undefined are left out."""
    diff = np.abs(atlas.Zstd[c] - atlas.Zstd[q])
    usable = [i for i in range(len(FEATURES))
              if not (atlas.imputed[q, i] or atlas.imputed[c, i])]
    closest = sorted(usable, key=lambda i: (diff[i], FEATURES[i]))[:3]
    largest = sorted(usable, key=lambda i: (-diff[i], FEATURES[i]))[:2]
    return ([_item(atlas, q, c, i) for i in closest],
            [_item(atlas, q, c, i) for i in largest])


def find_similar(atlas, fighter_id, n=10, same_cluster=False, division=None,
                 category=None, min_minutes=None):
    """Closest fighters to `fighter_id` among the model cohort.

    same_cluster: only candidates in the same style group.
    division: None = any division, "same" = the division the fighter fought in most,
              or an explicit division name.
    category: None = everyone, "same", "men", or "women" (women = fought in at least
              one Women's division).
    min_minutes: only candidates with at least this much recorded fight time.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError("n must be a whole number between 1 and 100")
    if division is not None and not isinstance(division, str):
        raise ValueError("division must be None, 'same', or a division name")
    if category not in (None, "same", "men", "women"):
        raise ValueError("category must be None, 'same', 'men', or 'women'")
    if min_minutes is not None and (isinstance(min_minutes, bool)
                                    or not isinstance(min_minutes, (int, float))
                                    or min_minutes < 0):
        raise ValueError("min_minutes must be a non-negative number")
    q = atlas.row_of.get(fighter_id)
    if q is None:
        raise FighterNotFound(fighter_id)

    prof = atlas.profiles
    mask = atlas.in_cohort.copy()
    mask[q] = False                                   # never match a fighter to themselves
    filters = {"same_cluster": bool(same_cluster), "division": None,
               "category": None, "min_minutes": min_minutes}
    if same_cluster:
        mask &= atlas.cluster == atlas.cluster[q]
    if division is not None:
        col = _div_col(atlas)
        target = prof.at[q, col] if division == "same" else division
        mask &= (prof[col] == target).to_numpy()
        filters["division"] = target
    if category is not None:
        if "women_division_fighter" not in prof:
            raise ValueError("this profile file has no category information; rebuild profiles")
        target = _category(atlas, q) if category == "same" else category
        mask &= (prof["women_division_fighter"].to_numpy() == (target == "women"))
        filters["category"] = target
    if min_minutes is not None:
        mask &= (prof["minutes"] >= min_minutes).to_numpy()

    pool = np.flatnonzero(mask)
    meta = atlas.artifact["metadata"]
    out = {
        "query": _query_summary(atlas, q),
        "comparison_cohort": {
            "definition": meta["cohort"]["definition"],
            "size": meta["cohort"]["size"],
            "candidates_after_filters": int(len(pool)),
            "filters": filters,
            "data_through": (meta.get("source_snapshot") or {}).get("date_max"),
        },
        "distance_note": "Lower distance means more similar.",
        "index_note": INDEX_NOTE,
        "limits": LIMITS,
        "results": [],
        "empty_reason": None,
    }
    if len(pool) == 0:
        out["empty_reason"] = ("No candidates match these filters. Loosen the style-group, "
                               "division, category, or minimum-minutes filter to see more fighters.")
        return out

    d = np.linalg.norm(atlas.Z[pool] - atlas.Z[q], axis=1)
    ids = prof["fighter_id"].to_numpy()[pool]
    order = np.lexsort((ids, d))[:n]                   # ties broken by fighter id
    d_ref = atlas.artifact["d_ref"]
    for rank, j in enumerate(order, start=1):
        c = int(pool[j])
        closest, largest = _explain(atlas, q, c)
        p = prof.iloc[c]
        out["results"].append({
            "rank": rank,
            "fighter_id": p["fighter_id"],
            "fighter_name": p["fighter_name"],
            "weight_class": p[_div_col(atlas)],
            "most_recent_weight_class": p["weight_class"],
            "distance": round(float(d[j]), 3),
            "similarity_index": round(float(100 * np.exp(-d[j] / d_ref)), 1),
            "group": _group_name(atlas, c),
            "same_group": bool(atlas.cluster[c] == atlas.cluster[q]),
            "fights": int(p["fights"]),
            "minutes": round(float(p["minutes"]), 1),
            "last_fight": str(p["last_fight"].date()),
            "undefined_features": _imputed_features(atlas, c),
            "closest_characteristics": closest,
            "largest_differences": largest,
        })
    return out


def search_fighters(atlas, text, limit=20):
    """Name search. Includes low-evidence fighters (flagged) so nobody is hidden."""
    text = (text or "").strip().lower()
    if len(text) < 2:
        return []
    p = atlas.profiles
    names = p["fighter_name"].fillna("").str.lower()
    hit = p[names.str.contains(text, regex=False)].copy()
    hit["_starts"] = names[hit.index].str.startswith(text)
    hit = hit.sort_values(["_starts", "minutes", "fighter_id"],
                          ascending=[False, False, True]).head(limit)
    return [{
        "fighter_id": r["fighter_id"], "fighter_name": r["fighter_name"],
        "weight_class": r[_div_col(atlas)], "fights": int(r["fights"]),
        "in_model_cohort": bool(atlas.in_cohort[atlas.row_of[r["fighter_id"]]]),
    } for _, r in hit.iterrows()]


def _fmt_item(it):
    q, m = it["query_value"], it["match_value"]
    return f"{it['label']}: {q} vs {m} ({it['unit']})"


def format_result(res):
    q = res["query"]
    recent = ("" if q["most_recent_weight_class"] == q["weight_class"]
              else f"; most recent: {q['most_recent_weight_class']}")
    lines = [
        f"{q['fighter_name']} ({q['weight_class']}{recent}) - {q['fights']} fights, "
        f"{q['minutes']} min, {q['first_fight']} to {q['last_fight']}",
        f"Style group: {q['group']}",
    ]
    if q["evidence_warning"]:
        lines.append("WARNING: " + q["evidence_warning"])
    if q["undefined_features"]:
        lines.append("Undefined (median-filled): " + ", ".join(q["undefined_features"]))
    c = res["comparison_cohort"]
    lines += [f"Compared with {c['candidates_after_filters']} of {c['size']} model-cohort "
              f"fighters (data through {c['data_through']}); filters: {c['filters']}",
              res["distance_note"], ""]
    if res["empty_reason"]:
        lines.append(res["empty_reason"])
    for r in res["results"]:
        lines.append(f"{r['rank']:>2}. {r['fighter_name']} ({r['weight_class']})  "
                     f"distance {r['distance']}  index {r['similarity_index']}  "
                     f"same group: {'yes' if r['same_group'] else 'no'}  "
                     f"[{r['fights']} fights, {r['minutes']} min]")
        lines.append("      closest:   " + "; ".join(_fmt_item(i) for i in r["closest_characteristics"]))
        lines.append("      differs:   " + "; ".join(_fmt_item(i) for i in r["largest_differences"]))
    lines += ["", res["index_note"], res["limits"]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Find fighters with similar styles")
    parser.add_argument("name")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--same-cluster", action="store_true")
    parser.add_argument("--division", default=None,
                        help="'same' for the fighter's main division, or a division name")
    parser.add_argument("--category", default=None, choices=["same", "men", "women"])
    parser.add_argument("--min-minutes", type=float, default=None)
    args = parser.parse_args()
    atlas = load_atlas()
    hits = search_fighters(atlas, args.name)
    if not hits:
        print(f"No fighter found matching '{args.name}'.")
        return
    if len(hits) > 1:
        print("Matches (using the first):",
              "; ".join(f"{h['fighter_name']} ({h['weight_class']})" for h in hits[:5]), "\n")
    res = find_similar(atlas, hits[0]["fighter_id"], n=args.n,
                       same_cluster=args.same_cluster, division=args.division,
                       category=args.category, min_minutes=args.min_minutes)
    print(format_result(res))


if __name__ == "__main__":
    main()
