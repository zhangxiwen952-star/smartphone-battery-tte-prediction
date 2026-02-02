# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd

import task3_common as tc


def main():
    parser = tc.parse_args_common()
    parser.add_argument("--use_temp", action="store_true", help="Enable temperature correction in this OAT run")
    parser.add_argument("--use_soh", action="store_true", help="Enable SOH correction in this OAT run")
    parser.add_argument("--pct", type=float, default=20.0, help="Perturbation percent for OAT (default 20%)")
    args = parser.parse_args()

    outdir = tc.ensure_dir(args.outdir)

    devices, sessions, samples = tc.load_excel(args.excel)
    scenario_df, sample_df = tc.preprocess(devices, sessions, samples)

    power0 = tc.PowerRegressor(model_type="linear")
    m0 = power0.fit(sample_df)

    E0 = tc.estimate_effective_capacity_wh(scenario_df, sample_df)
    options = tc.SimulationOptions(extend_mode=args.extend_mode, max_extend_hours=args.max_extend_hours)

    temp_model0 = tc.TempModel() if args.use_temp else None
    soh_model0 = tc.SoHModel(enabled=True) if args.use_soh else None

    base_eval = tc.evaluate_all_scenarios(
        scenario_df, sample_df, power0, E0,
        temp_model=temp_model0, soh_model=soh_model0,
        residual_power_w=0.0, options=options, max_scenarios=args.max_scenarios
    )
    base_mean_T = float(base_eval["t_empty_hours"].mean())
    pct = float(args.pct) / 100.0

    sens_rows = []

    def run_variant(name, power_model, E_eff, residual_w, temp_model, soh_model):
        ev = tc.evaluate_all_scenarios(
            scenario_df, sample_df, power_model, E_eff,
            temp_model=temp_model, soh_model=soh_model,
            residual_power_w=residual_w, options=options, max_scenarios=args.max_scenarios
        )
        mean_T = float(ev["t_empty_hours"].mean())
        delta_pct = (mean_T - base_mean_T) / base_mean_T * 100.0
        sens_rows.append({
            "parameter": name,
            "mean_t_empty_hours": mean_T,
            "delta_percent": delta_pct,
            "abs_delta_percent": abs(delta_pct)
        })

    # Capacity
    run_variant(f"E_eff_wh +{args.pct:.0f}%", power0, E0*(1+pct), 0.0, temp_model0, soh_model0)
    run_variant(f"E_eff_wh -{args.pct:.0f}%", power0, E0*(1-pct), 0.0, temp_model0, soh_model0)

    mean_power = float(sample_df["estimated_power_w"].mean())
    residual = mean_power * pct
    run_variant(f"residual_power +{args.pct:.0f}%meanP", power0, E0, residual, temp_model0, soh_model0)
    run_variant(f"residual_power -{args.pct:.0f}%meanP", power0, E0, -residual, temp_model0, soh_model0)

    # intercept
    p_int_up = power0.clone_with_scaled_params({}, intercept_multiplier=(1+pct))
    p_int_dn = power0.clone_with_scaled_params({}, intercept_multiplier=(1-pct))
    run_variant(f"intercept +{args.pct:.0f}%", p_int_up, E0, 0.0, temp_model0, soh_model0)
    run_variant(f"intercept -{args.pct:.0f}%", p_int_dn, E0, 0.0, temp_model0, soh_model0)

    # coefficients
    for feat in power0.feature_columns:
        up = power0.clone_with_scaled_params({feat: 1+pct}, intercept_multiplier=1.0)
        dn = power0.clone_with_scaled_params({feat: 1-pct}, intercept_multiplier=1.0)
        run_variant(f"coef[{feat}] +{args.pct:.0f}%", up, E0, 0.0, temp_model0, soh_model0)
        run_variant(f"coef[{feat}] -{args.pct:.0f}%", dn, E0, 0.0, temp_model0, soh_model0)

    # temperature slopes if enabled
    if args.use_temp:
        base_tm = tc.TempModel()
        tm_up = tc.TempModel(cold_slope=base_tm.cold_slope*(1+pct), hot_slope=base_tm.hot_slope*(1+pct), min_factor=base_tm.min_factor)
        tm_dn = tc.TempModel(cold_slope=base_tm.cold_slope*(1-pct), hot_slope=base_tm.hot_slope*(1-pct), min_factor=base_tm.min_factor)
        run_variant(f"temp_slopes +{args.pct:.0f}%", power0, E0, 0.0, tm_up, soh_model0)
        run_variant(f"temp_slopes -{args.pct:.0f}%", power0, E0, 0.0, tm_dn, soh_model0)

    sens_df = pd.DataFrame(sens_rows)
    tc.save_csv(sens_df, os.path.join(outdir, "exp04_oat_sensitivity.csv"))
    tc.plot_tornado(
        sens_df.groupby("parameter", as_index=False)["delta_percent"].mean().assign(
            abs_delta_percent=lambda d: d["delta_percent"].abs()
        ),
        out_path=os.path.join(outdir, "exp04_oat_tornado.png"),
        title="OAT sensitivity (mean time-to-empty)"
    )

    tc.save_json({
        "baseline_mean_t_empty_hours": base_mean_T,
        "power_model_metrics": m0,
        "E_eff_wh": E0,
        "features": power0.feature_columns,
        "use_temp": bool(args.use_temp),
        "use_soh": bool(args.use_soh),
        "pct": float(args.pct),
    }, os.path.join(outdir, "exp04_oat_meta.json"))

    print("Saved exp04 results to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
