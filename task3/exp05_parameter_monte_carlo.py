# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import task3_common as tc


def main():
    parser = tc.parse_args_common()
    parser.add_argument("--n", type=int, default=300, help="Number of Monte Carlo samples")
    parser.add_argument("--range", type=float, default=10.0, help="±range percent for multipliers (default 10)")
    parser.add_argument("--use_temp", action="store_true", help="Enable temperature correction and sample its slopes")
    parser.add_argument("--use_soh", action="store_true", help="Enable SOH correction")
    args = parser.parse_args()

    outdir = tc.ensure_dir(args.outdir)

    devices, sessions, samples = tc.load_excel(args.excel)
    scenario_df, sample_df = tc.preprocess(devices, sessions, samples)

    power0 = tc.PowerRegressor(model_type="linear")
    m0 = power0.fit(sample_df)

    E0 = tc.estimate_effective_capacity_wh(scenario_df, sample_df)
    options = tc.SimulationOptions(extend_mode=args.extend_mode, max_extend_hours=args.max_extend_hours)

    mean_power = float(sample_df["estimated_power_w"].mean())
    r = float(args.range) / 100.0

    scenario_ids = list(scenario_df["scenario_id"].unique())
    if args.max_scenarios is not None:
        scenario_ids = scenario_ids[:int(args.max_scenarios)]

    T = np.zeros((len(scenario_ids), int(args.n)), dtype=float)
    rng = np.random.default_rng(20260131)

    for j in range(int(args.n)):
        E_mult = rng.uniform(1-r, 1+r)
        intercept_mult = rng.uniform(1-r, 1+r)
        residual_w = rng.uniform(-r, r) * mean_power

        coef_mults = {feat: float(rng.uniform(1-r, 1+r)) for feat in power0.feature_columns}
        power_j = power0.clone_with_scaled_params(coef_mults, intercept_multiplier=intercept_mult)

        temp_model = None
        if args.use_temp:
            base_tm = tc.TempModel()
            s_mult = rng.uniform(1-r, 1+r)
            temp_model = tc.TempModel(
                cold_slope=base_tm.cold_slope*s_mult,
                hot_slope=base_tm.hot_slope*s_mult,
                min_factor=base_tm.min_factor
            )

        soh_model = tc.SoHModel(enabled=True) if args.use_soh else None

        ev = tc.evaluate_all_scenarios(
            scenario_df[scenario_df["scenario_id"].isin(scenario_ids)],
            sample_df,
            power_model=power_j,
            E_eff_wh=E0*E_mult,
            temp_model=temp_model,
            soh_model=soh_model,
            residual_power_w=residual_w,
            options=options,
            max_scenarios=None
        ).set_index("scenario_id")

        for i, sid in enumerate(scenario_ids):
            T[i, j] = float(ev.loc[sid, "t_empty_hours"]) if sid in ev.index else np.nan

    rows = []
    for i, sid in enumerate(scenario_ids):
        x = T[i, :]
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        rows.append({
            "scenario_id": sid,
            "mean": float(np.mean(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "std": float(np.std(x)),
        })
    stats_df = pd.DataFrame(rows)
    tc.save_csv(stats_df, os.path.join(outdir, "exp05_mc_per_scenario_stats.csv"))

    overall_mean = np.nanmean(T, axis=0)
    plt.figure(figsize=(8, 4.5))
    plt.hist(overall_mean[np.isfinite(overall_mean)], bins=30)
    plt.xlabel("Mean time-to-empty across scenarios (hours)")
    plt.ylabel("Count")
    plt.title("Monte Carlo distribution of mean time-to-empty")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "exp05_mc_overall_mean_hist.png"), dpi=200)
    plt.close()

    tc.save_json({
        "power_model_metrics": m0,
        "E_eff_wh": E0,
        "features": power0.feature_columns,
        "n_samples": int(args.n),
        "range_percent": float(args.range),
        "use_temp": bool(args.use_temp),
        "use_soh": bool(args.use_soh),
        "overall_mean_mean": float(np.nanmean(overall_mean)),
        "overall_mean_p05": float(np.nanquantile(overall_mean, 0.05)),
        "overall_mean_p95": float(np.nanquantile(overall_mean, 0.95)),
    }, os.path.join(outdir, "exp05_mc_meta.json"))

    print("Saved exp05 results to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
