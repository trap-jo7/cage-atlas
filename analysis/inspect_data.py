import pandas as pd
from pathlib import Path

RAW = Path("data/raw")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def mmss_to_seconds(s):
    try:
        m, sec = str(s).strip().split(":")
        m, sec = int(m), int(sec)
        if m < 0 or not 0 <= sec < 60:
            return float("nan")
        return m * 60 + sec
    except Exception:
        return float("nan")


# ---------------- UFC ----------------
ufc = pd.read_csv(RAW / "ufc-stats-complete-dataset" / "clean_ufc_dataset.csv")

section("UFC: shape and dates")
print(ufc.shape)
dates = pd.to_datetime(ufc["event_date"], errors="coerce")
print("date range:", dates.min(), "to", dates.max(), "| unparseable dates:", dates.isna().sum())
print("duplicate fight_url:", ufc["fight_url"].duplicated().sum())

section("UFC: missing values per column")
print(ufc.isna().sum()[lambda s: s > 0].sort_values(ascending=False))

section("UFC: weight classes")
print(ufc["weight_class"].value_counts(dropna=False))

section("UFC: methods (top 15)")
print(ufc["method"].value_counts(dropna=False).head(15))

section("UFC: round_num")
print(ufc["round_num"].value_counts(dropna=False).sort_index())

section("UFC: fight duration")
sec = ufc["time"].map(mmss_to_seconds)
print("unparseable times:", sec.isna().sum())
total = (pd.to_numeric(ufc["round_num"], errors="coerce") - 1) * 300 + sec
print(total.describe())
print("fights with duration <= 0 or missing:", (total.fillna(0) <= 0).sum())
print("time values over 5:00:", (sec > 300).sum())

section("UFC: stats sanity")
for side in ("f1", "f2"):
    landed = ufc[f"{side}_Sig_str_landed"]
    att = ufc[f"{side}_Sig_str_attempted"]
    print(side, "sig landed > attempted:", (landed > att).sum(),
          "| both zero:", ((landed == 0) & (att == 0)).sum())
    tl = ufc[f"{side}_Td_landed"]
    ta = ufc[f"{side}_Td_attempted"]
    print(side, "td landed > attempted:", (tl > ta).sum())
    print(side, "unparseable control times:", ufc[f"{side}_Ctrl"].map(mmss_to_seconds).isna().sum())
both_zero = ((ufc["f1_Sig_str_attempted"] == 0) & (ufc["f2_Sig_str_attempted"] == 0)).sum()
print("fights where both fighters have zero sig-str attempts:", both_zero)

section("UFC: fighters and evidence thresholds")
d = pd.DataFrame({"f1": ufc["fighter_1_url"], "f2": ufc["fighter_2_url"], "mins": total / 60})
mins = pd.concat([
    d[["f1", "mins"]].set_axis(["url", "mins"], axis=1),
    d[["f2", "mins"]].set_axis(["url", "mins"], axis=1),
]).dropna()
agg = mins.groupby("url")["mins"].agg(["count", "sum"])
print("unique fighters:", len(agg))
print("fighters with >= 3 fights:", (agg["count"] >= 3).sum())
print("eligible (>= 3 fights and >= 15 minutes):", ((agg["count"] >= 3) & (agg["sum"] >= 15)).sum())

# ---------------- PFL and Bellator ----------------
for name, path in [
    ("PFL", "pfl-complete-dataset/clean_sherdog_pfl.csv"),
    ("Bellator", "bellator-mma-complete-dataset/clean_sherdog_bellator.csv"),
]:
    df = pd.read_csv(RAW / path)
    section(f"{name}: shape, status, dates")
    print(df.shape)
    print(df["event_status"].value_counts(dropna=False))
    dts = pd.to_datetime(df["event_date"], errors="coerce", utc=True)
    print("date range:", dts.min(), "to", dts.max())
    print("fighter_1 results:", df["fighter_1_result"].value_counts(dropna=False).to_dict())
    print(df["method"].value_counts(dropna=False).head(12))
    print("stat-like columns:", [c for c in df.columns
                                 if any(k in c.lower() for k in ("sig_str", "td_", "ctrl", "kd"))])
    print("duplicate rows (event + fight_name):", df.duplicated(["event_name", "fight_name"]).sum())