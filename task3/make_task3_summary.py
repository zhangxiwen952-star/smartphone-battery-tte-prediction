import os, json
import pandas as pd

BASE = "task3_results"

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

rows = []

# exp01
p = os.path.join(BASE, "exp01_temperature", "exp01_temperature_summary.json")
if os.path.exists(p):
    j = read_json(p)
    rows.append({"exp":"temperature", "delta_mean_%": j["delta_mean_percent"], "E_eff_wh": j["E_eff_wh"]})

# exp02
p = os.path.join(BASE, "exp02_soh", "exp02_soh_summary.json")
if os.path.exists(p):
    j = read_json(p)
    rows.append({"exp":"soh", "delta_mean_%": j["delta_mean_percent"], "E_eff_wh": j["E_eff_wh"]})

# exp04 (OAT)
p = os.path.join(BASE, "exp04_oat", "exp04_oat_sensitivity.csv")
if os.path.exists(p):
    df = pd.read_csv(p).sort_values("abs_delta_percent", ascending=False).head(10)
    df.to_csv(os.path.join(BASE, "task3_top10_oat.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame(rows).to_csv(os.path.join(BASE, "task3_assumption_summary.csv"), index=False, encoding="utf-8-sig")
print("Saved:", os.path.join(BASE, "task3_assumption_summary.csv"), "and task3_top10_oat.csv")
