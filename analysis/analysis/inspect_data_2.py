import json
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def mmss(s):
    try:
        m, sec = str(s).strip().split(":")
        m, sec = int(m), int(sec)
        return m * 60 + sec if m >= 0 and 0 <= sec < 60 else float("nan")
    except Exception:
        return float("nan")


# 1. JSON files that came in the zips
for p in sorted(RAW.glob("*/*.json")):
    section(f"JSON: {p.name}")
    with open(p) as f:
        data = json.load(f)
    print("type:", type(data).__name__, "| length:", len(data))
    if isinstance(data, list) and data:
        print("first item:", json.dumps(data[0], indent=2)[:800])
    elif isinstance(data, dict) and data:
        keys = list(data.keys())
        print("first keys:", keys[:10])
        print("first value:", json.dumps(data[keys[0]], indent=2)[:800])

# 2. UFC oddities
ufc = pd.read_csv(RAW / "ufc-stats-complete-dataset" / "clean_ufc_dataset.csv")
year = pd.to_datetime(ufc["event_date"]).dt.year

section("UFC: what do unreadable control-time values look like, and when?")
mask = ufc["f1_Ctrl"].map(mmss).isna()
print(ufc.loc[mask, "f1_Ctrl"].value_counts(dropna=False).head())
print(year[mask].value_counts().sort_index().to_string())

section("UFC: fights with a time over 5:00")
sec = ufc["time"].map(mmss)
long_ = ufc[sec > 300]
print(long_[["event_date", "fighter_1", "fighter_2", "round_num", "time"]].head(15).to_string())
print("round_num among these:", long_["round_num"].value_counts().to_dict())

section("UFC: fights where both fighters have zero sig-str attempts")
bz = ufc[(ufc["f1_Sig_str_attempted"] == 0) & (ufc["f2_Sig_str_attempted"] == 0)]
print(bz[["event_date", "fighter_1", "fighter_2", "method", "round_num", "time"]].to_string())

section("UFC: fights with recorded strike stats, per year")
print(year[ufc["f1_Sig_str_attempted"].notna()].value_counts().sort_index().to_string())

# 3. PFL rows dated after the UFC snapshot
section("PFL: rows dated after 2026-05-16")
pfl = pd.read_csv(RAW / "pfl-complete-dataset" / "clean_sherdog_pfl.csv")
d = pd.to_datetime(pfl["event_date"], utc=True, errors="coerce")
late = pfl[d > pd.Timestamp("2026-05-16", tz="UTC")]
print("count:", len(late))
print(late[["event_date", "event_status", "fighter_1", "fighter_1_result", "method"]].head(10).to_string())