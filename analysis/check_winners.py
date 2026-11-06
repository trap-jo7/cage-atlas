import json
import pandas as pd

b = pd.read_parquet("data/processed/ufc_bouts.parquet")
print("winner_side counts:", b["winner_side"].value_counts().to_dict())

odd = b[b["winner_side"] == "none"]
print("\nwinner_name values on the odd rows:")
print(odd["winner_name"].value_counts(dropna=False).head(10))
print("\nmethods on the odd rows:")
print(odd["method"].value_counts().head(10))
print("\nsample odd rows:")
print(odd[["event_date", "fighter_1", "fighter_2", "method", "winner_name"]].head(8).to_string())

with open("data/raw/ufc-stats-complete-dataset/win_and_title_data-ufc.json") as f:
    meta = json.load(f)
extra = [k for k in meta if k not in set(b["fight_url"])]
print(f"\nJSON entries with no CSV row: {len(extra)}")
for k in extra[:13]:
    v = meta[k]
    print(v.get("event_date"), v.get("fighter_1"), "vs", v.get("fighter_2"), "| winner:", v.get("winner"))