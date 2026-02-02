# -*- coding: utf-8 -*-


import os
import subprocess
import sys


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=str, required=True)
    p.add_argument("--outdir", type=str, default="task3_results")
    p.add_argument("--max_scenarios", type=int, default=None)
    p.add_argument("--extend_mode", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--max_extend_hours", type=float, default=48.0)
    p.add_argument("--mc_n", type=int, default=300)
    p.add_argument("--mc_range", type=float, default=10.0)
    p.add_argument("--oat_pct", type=float, default=20.0)
    args = p.parse_args()

    base = os.path.abspath(args.outdir)
    os.makedirs(base, exist_ok=True)

    common_flags = [
        "--excel", os.path.abspath(args.excel),
        "--max_extend_hours", str(args.max_extend_hours),
        "--extend_mode", args.extend_mode,
    ]
    if args.max_scenarios is not None:
        common_flags += ["--max_scenarios", str(args.max_scenarios)]

    run([sys.executable, "exp01_assumption_temperature.py", "--outdir", os.path.join(base, "exp01_temperature"), *common_flags])
    run([sys.executable, "exp02_assumption_soh.py", "--outdir", os.path.join(base, "exp02_soh"), *common_flags])
    run([sys.executable, "exp03_assumption_feature_ablation.py", "--outdir", os.path.join(base, "exp03_feature_ablation"), *common_flags])
    run([sys.executable, "exp04_parameter_oat_sensitivity.py", "--outdir", os.path.join(base, "exp04_oat"), *common_flags, "--pct", str(args.oat_pct)])
    run([sys.executable, "exp05_parameter_monte_carlo.py", "--outdir", os.path.join(base, "exp05_mc"), *common_flags, "--n", str(args.mc_n), "--range", str(args.mc_range)])
    run([sys.executable, "exp06_usage_fluctuations.py", "--outdir", os.path.join(base, "exp06_fluctuations")])

    print("\nAll Task 3 experiments completed.")
    print("Results folder:", base)


if __name__ == "__main__":
    main()
