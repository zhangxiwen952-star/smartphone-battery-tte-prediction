# -*- coding: utf-8 -*-

import os
import pandas as pd

import task3_common as tc


def main():
    parser = tc.parse_args_common()
    args = parser.parse_args()

    outdir = tc.ensure_dir(args.outdir)

    devices, sessions, samples = tc.load_excel(args.excel)
    scenario_df, sample_df = tc.preprocess(devices, sessions, samples)

    options = tc.SimulationOptions(extend_mode=args.extend_mode, max_extend_hours=args.max_extend_hours)

    base_power = tc.PowerRegressor(model_type="linear")
    base_metrics = base_power.fit(sample_df)
    base_feats = list(base_power.feature_columns)

    E_eff = tc.estimate_effective_capacity_wh(scenario_df, sample_df)

    base_eval = tc.evaluate_all_scenarios(
        scenario_df, sample_df, base_power, E_eff,
        temp_model=None, soh_model=None, residual_power_w=0.0,
        options=options, max_scenarios=args.max_scenarios
    )
    base_sum = tc.summarize(base_eval)

    rows = [{
        "variant": "baseline_all_features",
        "dropped_feature": "",
        "power_r2": base_metrics["r2"],
        "power_rmse_w": base_metrics["rmse"],
        **base_sum
    }]

    for drop in base_feats:
        feats = [f for f in base_feats if f != drop]
        if not feats:
            continue
        power = tc.PowerRegressor(model_type="linear")
        m = power.fit(sample_df, features=feats)
        ev = tc.evaluate_all_scenarios(
            scenario_df, sample_df, power, E_eff,
            temp_model=None, soh_model=None, residual_power_w=0.0,
            options=options, max_scenarios=args.max_scenarios
        )
        s = tc.summarize(ev)
        rows.append({
            "variant": "drop_one_feature",
            "dropped_feature": drop,
            "power_r2": m["r2"],
            "power_rmse_w": m["rmse"],
            **s
        })

    result_df = pd.DataFrame(rows)
    if "mean_t_empty_hours" in result_df.columns:
        base_t = float(result_df.loc[result_df["variant"] == "baseline_all_features", "mean_t_empty_hours"].iloc[0])
        result_df["delta_t_empty_percent_vs_base"] = (result_df["mean_t_empty_hours"] - base_t) / base_t * 100.0

    tc.save_csv(result_df, os.path.join(outdir, "exp03_feature_ablation_summary.csv"))
    tc.save_json({
        "baseline_features": base_feats,
        "notes": "Drop-one-feature ablation; compare changes in mean time-to-empty and SOC errors."
    }, os.path.join(outdir, "exp03_feature_ablation_meta.json"))

    print("Saved exp03 results to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
