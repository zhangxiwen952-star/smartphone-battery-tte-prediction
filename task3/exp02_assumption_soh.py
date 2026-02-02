# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd

import task3_common as tc


def main():
    parser = tc.parse_args_common()
    args = parser.parse_args()

    outdir = tc.ensure_dir(args.outdir)

    devices, sessions, samples = tc.load_excel(args.excel)
    scenario_df, sample_df = tc.preprocess(devices, sessions, samples)

    power = tc.PowerRegressor(model_type="linear")
    metrics = power.fit(sample_df)

    E_eff = tc.estimate_effective_capacity_wh(scenario_df, sample_df)
    options = tc.SimulationOptions(extend_mode=args.extend_mode, max_extend_hours=args.max_extend_hours)

    base = tc.evaluate_all_scenarios(
        scenario_df, sample_df, power, E_eff,
        temp_model=None, soh_model=None,
        residual_power_w=0.0, options=options, max_scenarios=args.max_scenarios
    ).rename(columns={"t_empty_hours": "t_empty_base"})

    soh_model = tc.SoHModel(enabled=True)
    soh = tc.evaluate_all_scenarios(
        scenario_df, sample_df, power, E_eff,
        temp_model=None, soh_model=soh_model,
        residual_power_w=0.0, options=options, max_scenarios=args.max_scenarios
    ).rename(columns={"t_empty_hours": "t_empty_soh"})

    merged = pd.merge(
        base[["scenario_id", "soh_pct", "t_empty_base", "mae_soc", "rmse_soc"]],
        soh[["scenario_id", "t_empty_soh"]],
        on="scenario_id", how="inner"
    )
    merged["delta_hours"] = merged["t_empty_soh"] - merged["t_empty_base"]
    merged["delta_percent"] = np.where(
        merged["t_empty_base"] > 0, merged["delta_hours"] / merged["t_empty_base"] * 100.0, np.nan
    )

    tc.save_csv(merged, os.path.join(outdir, "exp02_soh_per_scenario.csv"))
    tc.save_json({
        "power_model_metrics": metrics,
        "E_eff_wh": E_eff,
        "baseline_summary": tc.summarize(base.rename(columns={"t_empty_base": "t_empty_hours"})),
        "soh_summary": tc.summarize(soh.rename(columns={"t_empty_soh": "t_empty_hours"})),
        "delta_mean_percent": float(np.nanmean(merged["delta_percent"])),
        "delta_median_percent": float(np.nanmedian(merged["delta_percent"])),
    }, os.path.join(outdir, "exp02_soh_summary.json"))

    print("Saved exp02 results to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
