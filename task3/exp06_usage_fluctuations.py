# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import task3_common as tc


def simulate_time_to_empty(power_series_w: np.ndarray, dt_seconds: float, soc0: float, E_eff_wh: float) -> float:
    soc = float(soc0)
    dt_h = dt_seconds / 3600.0
    for k, p in enumerate(power_series_w):
        p = max(float(p), 0.0)
        dsoc = -(p / E_eff_wh) * 100.0 * dt_h
        soc_next = soc + dsoc
        if soc_next <= 0:
            frac = soc / (soc - soc_next) if soc > 0 else 0.0
            frac = float(np.clip(frac, 0.0, 1.0))
            return (k + frac) * dt_seconds / 3600.0
        soc = soc_next
    return len(power_series_w) * dt_seconds / 3600.0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="task3_results")
    p.add_argument("--soc0", type=float, default=80.0)
    p.add_argument("--E", type=float, default=16.0, help="Effective capacity (Wh)")
    p.add_argument("--meanP", type=float, default=3.0, help="Mean power (W)")
    p.add_argument("--hours", type=float, default=24.0, help="Simulation horizon (hours)")
    p.add_argument("--dt", type=float, default=10.0, help="Time step (seconds)")
    p.add_argument("--burst_ratio", type=float, default=4.0, help="Spike-to-mean ratio for bursty pattern")
    args = p.parse_args()

    outdir = tc.ensure_dir(args.outdir)
    n = int(args.hours * 3600.0 / args.dt)
    t = np.arange(n) * args.dt / 3600.0

    meanP = float(args.meanP)
    E = float(args.E)

    smooth = np.full(n, meanP)

    spike = meanP * float(args.burst_ratio)
    spike_mask = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(20260131)
    spike_idx = rng.choice(n, size=max(1, n // 10), replace=False)
    spike_mask[spike_idx] = True
    low = (meanP * n - spike * spike_mask.sum()) / max(1, (~spike_mask).sum())
    bursty = np.where(spike_mask, spike, low)

    periodic = meanP * (1.0 + 0.5 * np.sin(2 * np.pi * t / 2.0))  # 2-hour period

    patterns = {"smooth": smooth, "bursty": bursty, "periodic": periodic}

    results = []
    for name, P in patterns.items():
        T = simulate_time_to_empty(P, args.dt, args.soc0, E)
        results.append({"pattern": name, "time_to_empty_hours": float(T)})

    df = pd.DataFrame(results)
    tc.save_csv(df, os.path.join(outdir, "exp06_fluctuations_time_to_empty.csv"))

    m = int(min(n, 6 * 3600.0 / args.dt))
    plt.figure(figsize=(10, 4.5))
    for name, P in patterns.items():
        plt.plot(t[:m], P[:m], label=name)
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (W)")
    plt.title("Usage fluctuation patterns (same mean power)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "exp06_fluctuations_patterns.png"), dpi=200)
    plt.close()

    tc.save_json({
        "soc0": args.soc0,
        "E_eff_wh": E,
        "mean_power_w": meanP,
        "dt_seconds": args.dt,
        "hours_horizon": args.hours,
        "burst_ratio": args.burst_ratio,
        "results": results
    }, os.path.join(outdir, "exp06_fluctuations_meta.json"))

    print("Saved exp06 results to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
