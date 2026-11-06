"""Style model: impute, scale, apply group weights, fit K-means (+ PCA for the map).

Run from the project root:
    python -m analysis.cageatlas.model evaluate        # compare K = 3..8, no model saved
    python -m analysis.cageatlas.model train --k 5     # fit the chosen K and save it
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files; never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

from .profiles import FEATURE_INFO, FEATURES

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
DIAG_DIR = MODEL_DIR / "diagnostics"

SEED = 42
N_INIT = 10
K_RANGE = range(3, 9)
STABILITY_SEEDS = [0, 1, 2, 3, 4]

# Share of the total squared weight given to each feature group.
GROUP_SHARES = {"striking": 0.45, "grappling": 0.35, "control": 0.20}
ALT_SHARES = {
    "equal_groups": {"striking": 1 / 3, "grappling": 1 / 3, "control": 1 / 3},
    "unweighted": None,  # every feature weight = 1
}

# (feature, direction) -> plain-language phrase used only when the numbers support it
PHRASES = {
    ("sig_landed_per_min", 1): "high strike volume",
    ("sig_landed_per_min", -1): "low strike volume",
    ("sig_absorbed_per_min", 1): "absorbs many strikes",
    ("sig_absorbed_per_min", -1): "absorbs few strikes",
    ("sig_accuracy", 1): "accurate striking",
    ("sig_accuracy", -1): "low striking accuracy",
    ("sig_defense", 1): "strong strike defense",
    ("sig_defense", -1): "weak strike defense",
    ("kd_per15", 1): "frequent knockdowns",
    ("kd_per15", -1): "few knockdowns",
    ("td_attempted_per15", 1): "many takedown attempts",
    ("td_attempted_per15", -1): "few takedown attempts",
    ("td_accuracy", 1): "efficient takedowns",
    ("td_accuracy", -1): "inefficient takedowns",
    ("td_defense", 1): "strong takedown defense",
    ("td_defense", -1): "weak takedown defense",
    ("ctrl_share", 1): "high control time",
    ("ctrl_share", -1): "little control time",
}
LABEL_MIN_ABS_Z = 0.5  # a trait must be at least this far from the cohort average


def feature_weights(features=FEATURES, shares=GROUP_SHARES):
    """Per-feature weights so each group's squared weights sum to its share."""
    if shares is None:
        return np.ones(len(features))
    group_of = [FEATURE_INFO[f]["group"] for f in features]
    counts = {g: group_of.count(g) for g in set(group_of)}
    return np.array([np.sqrt(shares[g] / counts[g]) for g in group_of])


def fit_space(X_raw):
    """Median-impute undefined values, then standardize. Returns fitted objects."""
    imputer = SimpleImputer(strategy="median").fit(X_raw)
    scaler = StandardScaler().fit(imputer.transform(X_raw))
    return imputer, scaler


def standardize(imputer, scaler, X_raw):
    return scaler.transform(imputer.transform(X_raw))


def load_cohort(window="career"):
    p = pd.read_parquet(PROC_DIR / f"ufc_profiles_{window}.parquet")
    return p[p["meets_threshold"]].reset_index(drop=True)


def kmeans_fit(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(Z)


def stability(Z, k):
    """Agreement (adjusted Rand index) between clusterings from different seeds."""
    labelings = [kmeans_fit(Z, k, s).labels_ for s in STABILITY_SEEDS]
    aris = [adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)]
    return float(np.mean(aris)), float(np.min(aris))


def median_pairwise_distance(Z):
    d = pairwise_distances(Z)
    return float(np.median(d[np.triu_indices_from(d, k=1)]))


def neighbor_overlap(Z_a, Z_b, n_probe=300, top=10, seed=SEED):
    """Average share of the top-N neighbors two spaces agree on (1.0 = identical)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z_a), size=min(n_probe, len(Z_a)), replace=False)

    def topk(Z):
        d = np.linalg.norm(Z[idx, None, :] - Z[None, :, :], axis=2)
        d[np.arange(len(idx)), idx] = np.inf  # a fighter is never their own neighbor
        return np.argsort(d, axis=1)[:, :top]

    a, b = topk(Z_a), topk(Z_b)
    return float(np.mean([len(set(x) & set(y)) / top for x, y in zip(a, b)]))


def describe_clusters(centroid_z):
    """Neutral group name, plus traits only where the centroid is clearly off-average."""
    names = {}
    for i, (cid, row) in enumerate(centroid_z.iterrows()):
        letter = chr(ord("A") + i)
        strong = row[row.abs() >= LABEL_MIN_ABS_Z].sort_values(key=abs, ascending=False)
        traits = [PHRASES[(f, 1 if z > 0 else -1)] for f, z in strong.head(2).items()]
        names[int(cid)] = f"Group {letter}: " + ", ".join(traits) if traits else (
            f"Group {letter}: no distinct trait"
        )
    return names


def outlier_review(Zstd):
    z = pd.DataFrame(Zstd, columns=FEATURES)
    return {
        "skew": {f: round(float(z[f].skew()), 2) for f in FEATURES},
        "max_abs_z": {f: round(float(z[f].abs().max()), 2) for f in FEATURES},
        "fighters_with_any_abs_z_over_4": int((z.abs() > 4).any(axis=1).sum()),
    }


def evaluate():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    X = cohort[FEATURES].to_numpy(dtype=float)
    imputer, scaler = fit_space(X)
    Zstd = standardize(imputer, scaler, X)
    Z = Zstd * feature_weights()

    rows = []
    for k in K_RANGE:
        km = kmeans_fit(Z, k)
        sizes = np.bincount(km.labels_)
        mean_ari, min_ari = stability(Z, k)
        rows.append(
            dict(k=k, silhouette=round(float(silhouette_score(Z, km.labels_)), 4),
                 inertia=round(float(km.inertia_), 1),
                 smallest_cluster=int(sizes.min()), largest_cluster=int(sizes.max()),
                 stability_mean_ari=round(mean_ari, 3),
                 stability_min_ari=round(min_ari, 3))
        )
    table = pd.DataFrame(rows)
    table.to_csv(DIAG_DIR / "k_evaluation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(table["k"], table["inertia"], marker="o")
    axes[0].set(title="Inertia (elbow)", xlabel="K", ylabel="inertia")
    axes[1].plot(table["k"], table["silhouette"], marker="o")
    axes[1].set(title="Silhouette", xlabel="K", ylabel="silhouette")
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "k_diagnostics.png", dpi=150)

    sensitivity = {}
    for name, shares in ALT_SHARES.items():
        Z_alt = Zstd * feature_weights(shares=shares)
        sensitivity[name] = round(neighbor_overlap(Z, Z_alt), 3)
    report = {
        "cohort_size": int(len(cohort)),
        "outlier_review": outlier_review(Zstd),
        "weight_sensitivity_top10_neighbor_overlap_vs_chosen_weights": sensitivity,
    }
    with open(DIAG_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Cohort: {len(cohort)} fighters\n")
    print(table.to_string(index=False))
    print("\n" + json.dumps(report, indent=2))
    print(f"\nSaved: {DIAG_DIR}/k_evaluation.csv, k_diagnostics.png, evaluation_report.json")


def train(k):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    X = cohort[FEATURES].to_numpy(dtype=float)
    weights = feature_weights()
    imputer, scaler = fit_space(X)
    Zstd = standardize(imputer, scaler, X)
    Z = Zstd * weights

    km = kmeans_fit(Z, k)
    labels = km.labels_
    pca = PCA(n_components=2, random_state=SEED).fit(Z)  # for the map only
    mean_ari, min_ari = stability(Z, k)
    imputed = np.isnan(X)

    centroid_z = pd.DataFrame(Zstd, columns=FEATURES).groupby(labels).mean()
    names = describe_clusters(centroid_z)
    sizes = np.bincount(labels)

    try:
        with open(PROC_DIR / "ufc_validation_report.json") as f:
            snap = json.load(f)
    except FileNotFoundError:
        snap = {}

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": "career",
        "k": k,
        "seed": SEED,
        "n_init": N_INIT,
        "cohort": {
            "definition": "UFC fighters, career window, at least 3 eligible fights and "
                          "15 minutes (eligible = fights from 2001-01-01 on with "
                          "complete stats and a valid duration)",
            "size": int(len(cohort)),
            "fighters_with_imputed_values": int(imputed.any(axis=1).sum()),
            "imputed_cells_by_feature": {f: int(imputed[:, i].sum())
                                         for i, f in enumerate(FEATURES)},
        },
        "source_snapshot": {key: snap.get(key) for key in
                            ("source", "rows", "date_min", "date_max", "built_at_utc")},
        "features": {f: FEATURE_INFO[f] for f in FEATURES},
        "group_shares": GROUP_SHARES,
        "weights": {f: round(float(w), 4) for f, w in zip(FEATURES, weights)},
        "missing_value_handling": "undefined rates (no attempts) are filled with the "
                                  "cohort median and flagged; never treated as zero",
        "scaling": "StandardScaler on median-imputed features; no clipping",
        "diagnostics": {
            "silhouette": round(float(silhouette_score(Z, labels)), 4),
            "inertia": round(float(km.inertia_), 1),
            "cluster_sizes": {names[i]: int(s) for i, s in enumerate(sizes)},
            "stability_mean_ari": round(mean_ari, 3),
            "stability_min_ari": round(min_ari, 3),
            "pca_explained_variance_ratio": [round(float(v), 4)
                                             for v in pca.explained_variance_ratio_],
            "reference_distance_median_pairwise": round(median_pairwise_distance(Z), 4),
        },
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "note": "Groups are exploratory style descriptions, not rankings.",
    }
    artifact = {
        "features": FEATURES,
        "weights": weights,
        "imputer": imputer,
        "scaler": scaler,
        "kmeans": km,
        "pca": pca,
        "cohort_ids": cohort["fighter_id"].tolist(),
        "labels": labels,
        "cluster_names": names,
        "pca_xy": pca.transform(Z),
        "d_ref": metadata["diagnostics"]["reference_distance_median_pairwise"],
        "metadata": metadata,
    }
    joblib.dump(artifact, MODEL_DIR / "ufc_career_model.joblib")
    with open(MODEL_DIR / "ufc_career_model_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    raw_means = cohort[FEATURES].groupby(labels).mean().round(3)
    raw_means.index = [names[i] for i in raw_means.index]
    comp = pd.crosstab(pd.Series(labels, name="cluster"), cohort["weight_class"])
    print(json.dumps(metadata["diagnostics"], indent=2))
    sil = metadata["diagnostics"]["silhouette"]
    if sil < 0.25:
        print("\nNOTE: silhouette is below 0.25, so the groups are not well separated; "
              "treat them as soft descriptions of a continuous spectrum.")
    print("\nAverage raw values per group:")
    print(raw_means.T.to_string())
    print("\nGroup sizes by most recent weight class:")
    print(comp.to_string())
    print(f"\nSaved: {MODEL_DIR}/ufc_career_model.joblib and ufc_career_model_meta.json")


def main():
    parser = argparse.ArgumentParser(description="CAGE ATLAS style model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate", help="compare K = 3..8 and save diagnostics")
    t = sub.add_parser("train", help="fit and save the model for a chosen K")
    t.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    evaluate() if args.command == "evaluate" else train(args.k)


if __name__ == "__main__":
    main()
