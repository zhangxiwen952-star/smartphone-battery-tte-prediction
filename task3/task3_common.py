# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score



def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_excel(excel_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the uploaded Excel schema:
      devices, sessions, samples

    If those sheets are not present, we fall back to old indices:
      sheet 1: scenario config
      sheet 2: samples
    """
    xl = pd.ExcelFile(excel_path)
    names = set(xl.sheet_names)

    if {"devices", "sessions", "samples"}.issubset(names):
        devices = pd.read_excel(excel_path, sheet_name="devices")
        sessions = pd.read_excel(excel_path, sheet_name="sessions")
        samples = pd.read_excel(excel_path, sheet_name="samples")
        return devices, sessions, samples

    # fallback (legacy)
    devices = pd.DataFrame()
    sessions = pd.read_excel(excel_path, sheet_name=1)
    samples = pd.read_excel(excel_path, sheet_name=2)
    return devices, sessions, samples


def _map_network_type_code(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    mapping = {
        "无": 0.0, "none": 0.0, "airplane": 0.0,
        "WiFi": 0.2, "wifi": 0.2,
        "4G": 0.5, "LTE": 0.5, "4g": 0.5,
        "5G": 1.0, "5g": 1.0,
    }
    return float(mapping.get(s, 0.0))


def preprocess(devices: pd.DataFrame, sessions: pd.DataFrame, samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce canonical scenario_df and sample_df from the uploaded schema.
    """
    scenario_df = sessions.copy()

    # Join SOH from devices if available
    if len(devices) > 0 and "device_id" in devices.columns and "device_id" in scenario_df.columns:
        join_cols = ["device_id"]
        if "battery_health_soh_pct" in devices.columns:
            scenario_df = scenario_df.merge(
                devices[["device_id", "battery_health_soh_pct"]],
                on="device_id",
                how="left"
            )
        else:
            scenario_df["battery_health_soh_pct"] = np.nan
    else:
        if "battery_health_soh_pct" not in scenario_df.columns:
            scenario_df["battery_health_soh_pct"] = np.nan

    # Standardize names
    if "scenario_name" not in scenario_df.columns and "场景名称" in scenario_df.columns:
        scenario_df["scenario_name"] = scenario_df["场景名称"]

    # Ensure required columns exist with reasonable defaults
    defaults = {
        "sampling_interval_s": 10.0,
        "ambient_temp_c_set": 25.0,
        "soc_start_pct": np.nan,
        "soc_end_pct": np.nan,
        "battery_health_soh_pct": 95.0,
    }
    for c, v in defaults.items():
        if c not in scenario_df.columns:
            scenario_df[c] = v
        scenario_df[c] = pd.to_numeric(scenario_df[c], errors="coerce").fillna(v)

    # SOC change
    scenario_df["soc_change_pct"] = (scenario_df["soc_start_pct"] - scenario_df["soc_end_pct"]).astype(float)

    # Sample-level
    sample_df = samples.copy()

    # Required columns mapping (if legacy CN columns exist)
    rename_map = {
        "t_秒": "t_s",
        "SOC真实值%": "soc_true_pct",
        "估计功耗W": "estimated_power_w",
        "亮度%": "brightness_pct",
        "CPU利用率%": "cpu_util_pct",
        "信号强度dBm": "signal_strength_dbm",
        "上行KB_间隔": "uplink_kb_per_interval",
        "下行KB_间隔": "downlink_kb_per_interval",
        "网络类型": "network_type",
        "定位服务": "location_service_01",
    }
    for k, v in rename_map.items():
        if k in sample_df.columns and v not in sample_df.columns:
            sample_df[v] = sample_df[k]

    # Numeric conversions
    for c in ["t_s", "soc_true_pct", "estimated_power_w", "brightness_pct", "cpu_util_pct",
              "uplink_kb_per_interval", "downlink_kb_per_interval", "signal_strength_dbm",
              "background_app_count_proxy"]:
        if c in sample_df.columns:
            sample_df[c] = pd.to_numeric(sample_df[c], errors="coerce")

    # Ensure columns exist
    if "t_s" not in sample_df.columns:
        sample_df["t_s"] = np.arange(len(sample_df)) * float(scenario_df["sampling_interval_s"].iloc[0] if len(scenario_df) else 10.0)
    if "soc_true_pct" not in sample_df.columns and "soc_display_pct" in sample_df.columns:
        sample_df["soc_true_pct"] = sample_df["soc_display_pct"]
    if "soc_true_pct" not in sample_df.columns:
        sample_df["soc_true_pct"] = np.nan

    if "estimated_power_w" not in sample_df.columns:
        raise ValueError("samples sheet must contain estimated_power_w (or 估计功耗W).")

    # Data rate (KB/s) using scenario sampling interval (usually 10s)
    if "uplink_kb_per_interval" in sample_df.columns and "downlink_kb_per_interval" in sample_df.columns:
        # Prefer per-sample scenario interval if present, else assume 10s
        if "scenario_id" in sample_df.columns and "scenario_id" in scenario_df.columns and "sampling_interval_s" in scenario_df.columns:
            interval_map = scenario_df.set_index("scenario_id")["sampling_interval_s"].to_dict()
            interval_s = sample_df["scenario_id"].map(interval_map).fillna(10.0).astype(float)
        else:
            interval_s = 10.0
        sample_df["data_rate_kb_s"] = (sample_df["uplink_kb_per_interval"].fillna(0.0) +
                                       sample_df["downlink_kb_per_interval"].fillna(0.0)) / interval_s
    else:
        sample_df["data_rate_kb_s"] = 0.0

    # Signal strength index (higher means worse signal, similar to your baseline idea: -dBm)
    if "signal_strength_dbm" in sample_df.columns:
        sample_df["signal_strength_index"] = (-sample_df["signal_strength_dbm"]).fillna(0.0)
    else:
        sample_df["signal_strength_index"] = 0.0

    # Network type code
    if "network_type" in sample_df.columns:
        sample_df["network_type_code"] = sample_df["network_type"].apply(_map_network_type_code).astype(float)
    else:
        sample_df["network_type_code"] = 0.0

    # GPS on/off
    if "location_service_01" in sample_df.columns:
        sample_df["gps_on"] = sample_df["location_service_01"].fillna(0).astype(int)
    else:
        sample_df["gps_on"] = 0

    # Fill NaNs for key numeric columns (use mean)
    numeric_cols = ["soc_true_pct", "estimated_power_w", "brightness_pct", "cpu_util_pct",
                    "data_rate_kb_s", "signal_strength_index", "network_type_code", "gps_on",
                    "background_app_count_proxy"]
    for c in numeric_cols:
        if c in sample_df.columns:
            if sample_df[c].isna().all():
                sample_df[c] = 0.0
            else:
                sample_df[c] = sample_df[c].fillna(sample_df[c].mean())

    return scenario_df, sample_df



@dataclass
class TempModel:
    """
    Temperature correction that scales effective capacity.

    factor(T) = 1 - cold_slope*(25-T) for T<=25
              = 1 - hot_slope*(T-25)  for T>25
    clipped to [min_factor, 1.0]
    """
    cold_slope: float = 0.008
    hot_slope: float = 0.002
    min_factor: float = 0.60

    def factor(self, temp_c: float) -> float:
        if temp_c <= 25:
            f = 1.0 - self.cold_slope * (25.0 - temp_c)
        else:
            f = 1.0 - self.hot_slope * (temp_c - 25.0)
        return float(np.clip(f, self.min_factor, 1.0))


@dataclass
class SoHModel:
    """
    SOH correction scaling effective capacity.
    """
    enabled: bool = True

    def factor(self, soh_percent: float) -> float:
        if not self.enabled:
            return 1.0
        if soh_percent is None or np.isnan(soh_percent):
            return 1.0
        return float(np.clip(soh_percent / 100.0, 0.50, 1.05))


class PowerRegressor:
    """
    Power consumption model wrapper.
    Baseline: LinearRegression. Optional: Ridge for stability.
    """
    def __init__(self, model_type: str = "linear", ridge_alpha: float = 1.0):
        self.model_type = model_type
        self.ridge_alpha = ridge_alpha
        self.model = None
        self.feature_columns: List[str] = []

    def fit(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, float]:
        if "estimated_power_w" not in df.columns:
            raise ValueError("Missing target column: estimated_power_w")

        if features is None:
            # Baseline-like features (plus optional background proxy)
            candidates = [
                "brightness_pct",
                "cpu_util_pct",
                "data_rate_kb_s",
                "signal_strength_index",
                "network_type_code",
                "gps_on",
            ]
            if "background_app_count_proxy" in df.columns:
                candidates.append("background_app_count_proxy")
            features = [c for c in candidates if c in df.columns]

        if not features:
            raise ValueError("No available features to train power model.")

        X = df[features].to_numpy()
        y = df["estimated_power_w"].to_numpy()

        if self.model_type == "ridge":
            self.model = Ridge(alpha=self.ridge_alpha)
        else:
            self.model = LinearRegression()

        self.model.fit(X, y)
        self.feature_columns = features

        y_pred = self.model.predict(X)
        r2 = float(r2_score(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        return {"r2": r2, "rmse": rmse}

    def predict(self, row: pd.Series) -> float:
        if self.model is None:
            raise RuntimeError("Power model not fitted.")
        x = np.array([[float(row.get(feat, 0.0)) for feat in self.feature_columns]])
        return float(self.model.predict(x)[0])

    def clone_with_scaled_params(self, coef_multipliers: Dict[str, float], intercept_multiplier: float = 1.0) -> "PowerRegressor":
        if self.model is None:
            raise RuntimeError("Power model not fitted.")
        if not hasattr(self.model, "coef_"):
            raise RuntimeError("Model does not expose coefficients.")
        import copy
        new = PowerRegressor(model_type=self.model_type, ridge_alpha=self.ridge_alpha)
        new.model = copy.deepcopy(self.model)
        new.feature_columns = list(self.feature_columns)

        coefs = np.array(new.model.coef_, dtype=float)
        for i, feat in enumerate(new.feature_columns):
            coefs[i] *= float(coef_multipliers.get(feat, 1.0))
        new.model.coef_ = coefs
        new.model.intercept_ = float(new.model.intercept_) * float(intercept_multiplier)
        return new


@dataclass
class SimulationOptions:
    dt_default_seconds: float = 10.0
    extend_mode: str = "last"  # "last" or "mean"
    max_extend_hours: float = 48.0


def estimate_effective_capacity_wh(scenario_df: pd.DataFrame, sample_df: pd.DataFrame) -> float:

    if "scenario_id" not in scenario_df.columns or "scenario_id" not in sample_df.columns:
        raise ValueError("Missing scenario_id in scenario_df or sample_df")

    dt_s_default = float(scenario_df["sampling_interval_s"].iloc[0]) if len(scenario_df) else 10.0

    capacities = []
    for sid in scenario_df["scenario_id"].unique():
        seg = sample_df[sample_df["scenario_id"] == sid].sort_values("t_s").reset_index(drop=True)
        if len(seg) < 2:
            continue
        dt_s = float(scenario_df.loc[scenario_df["scenario_id"] == sid, "sampling_interval_s"].iloc[0]) if "sampling_interval_s" in scenario_df.columns else dt_s_default
        total_energy_Wh = float(seg["estimated_power_w"].sum()) * (dt_s / 3600.0)
        soc_drop = float(seg["soc_true_pct"].iloc[0] - seg["soc_true_pct"].iloc[-1])
        if soc_drop > 0 and total_energy_Wh > 0:
            capacities.append(total_energy_Wh / (soc_drop / 100.0))

    if not capacities:
        return 16.0
    return float(np.mean(capacities))


def _get_scenario_temp(srow: pd.Series) -> float:
    for c in ["ambient_temp_c_set", "battery_temp_c", "Temperature", "temperature_c"]:
        if c in srow.index:
            try:
                return float(srow[c])
            except Exception:
                pass
    return 25.0


def _get_scenario_soh(srow: pd.Series) -> float:
    for c in ["battery_health_soh_pct", "soh", "SOH", "SOH%"]:
        if c in srow.index:
            try:
                return float(srow[c])
            except Exception:
                pass
    return 95.0


def simulate_soc(
    scenario_id: int,
    scenario_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    power_model: PowerRegressor,
    E_eff_wh: float,
    temp_model: Optional[TempModel] = None,
    soh_model: Optional[SoHModel] = None,
    residual_power_w: float = 0.0,
    options: Optional[SimulationOptions] = None
) -> Dict[str, object]:
    """
    Simulate SOC trajectory and time-to-empty for a scenario.
    Uses discrete integration consistent with continuous-time ODE.

    Return per-scenario metrics including time-to-empty and SOC error stats.
    """
    if options is None:
        options = SimulationOptions()

    seg = sample_df[sample_df["scenario_id"] == scenario_id].copy()
    if len(seg) < 2:
        return {"scenario_id": scenario_id, "ok": False, "reason": "insufficient samples"}

    seg = seg.sort_values("t_s").reset_index(drop=True)

    srow = scenario_df[scenario_df["scenario_id"] == scenario_id].iloc[0]
    temp_c = _get_scenario_temp(srow)
    soh = _get_scenario_soh(srow)

    temp_factor = temp_model.factor(temp_c) if temp_model is not None else 1.0
    soh_factor = soh_model.factor(soh) if soh_model is not None else 1.0

    E_adj = max(float(E_eff_wh) * temp_factor * soh_factor, 1e-6)

    t_s = seg["t_s"].to_numpy(dtype=float)
    t_h = t_s / 3600.0
    soc0 = float(seg["soc_true_pct"].iloc[0]) if "soc_true_pct" in seg.columns else 100.0
    soc_pred = [soc0]

    reached = False
    t_empty_h = None

    for i in range(1, len(seg)):
        dt_h = float((t_s[i] - t_s[i-1]) / 3600.0)
        if dt_h <= 0:
            dt_h = options.dt_default_seconds / 3600.0

        power = power_model.predict(seg.iloc[i]) + float(residual_power_w)
        if not np.isfinite(power):
            power = 0.0
        power = max(float(power), 0.0)
        dsoc = -(power / E_adj) * 100.0 * dt_h
        new_soc = soc_pred[-1] + dsoc

        if new_soc <= 0 and not reached:
            reached = True
            frac = soc_pred[-1] / (soc_pred[-1] - new_soc) if soc_pred[-1] > 0 else 0.0
            frac = float(np.clip(frac, 0.0, 1.0))
            t_empty_h = float(t_h[i-1] + frac * dt_h)
            soc_pred.append(0.0)
            break

        soc_pred.append(float(np.clip(new_soc, 0.0, 100.0)))

    # Extend beyond recorded window if needed (analytic, constant-power extension)
    if not reached:
        if options.extend_mode == "mean":
            ref_row = seg[power_model.feature_columns].mean(numeric_only=True)
        else:
            ref_row = seg.iloc[-1]

        power_const = power_model.predict(ref_row) + float(residual_power_w)
        if not np.isfinite(power_const):
            power_const = 0.0
        power_const = max(float(power_const), 0.0)

        cur_soc = float(soc_pred[-1])
        cur_t = float(t_h[min(len(t_h)-1, len(soc_pred)-1)])

        if power_const <= 1e-9:
            # cannot discharge further under this assumption; cap at max_extend_hours
            t_empty_h = float(cur_t + options.max_extend_hours)
            reached = False
        else:
            t_rem_h = (cur_soc / 100.0) * (E_adj / power_const)
            # apply cap
            t_rem_h = float(min(t_rem_h, options.max_extend_hours))
            t_empty_h = float(cur_t + t_rem_h)
            reached = True

    # Error on observed points
    soc_pred_arr = np.array(soc_pred, dtype=float)
    mae = None
    rmse = None
    if "soc_true_pct" in seg.columns and seg["soc_true_pct"].notna().any():
        pred_times = np.linspace(t_h[0], t_h[min(len(t_h)-1, len(soc_pred_arr)-1)], num=len(soc_pred_arr))
        pred_at_obs = np.interp(t_h, pred_times, soc_pred_arr[:len(pred_times)])
        soc_actual = seg["soc_true_pct"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(soc_actual - pred_at_obs)))
        rmse = float(np.sqrt(mean_squared_error(soc_actual, pred_at_obs)))

    return {
        "scenario_id": scenario_id,
        "ok": True,
        "temp_c": float(temp_c),
        "soh_pct": float(soh),
        "temp_factor": float(temp_factor),
        "soh_factor": float(soh_factor),
        "E_eff_wh": float(E_eff_wh),
        "E_adj_wh": float(E_adj),
        "t_empty_hours": float(t_empty_h) if t_empty_h is not None else None,
        "reached_empty": bool(reached),
        "mae_soc": mae,
        "rmse_soc": rmse,
    }


def evaluate_all_scenarios(
    scenario_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    power_model: PowerRegressor,
    E_eff_wh: float,
    temp_model: Optional[TempModel] = None,
    soh_model: Optional[SoHModel] = None,
    residual_power_w: float = 0.0,
    options: Optional[SimulationOptions] = None,
    max_scenarios: Optional[int] = None,
) -> pd.DataFrame:
    sids = list(scenario_df["scenario_id"].unique())
    if max_scenarios is not None:
        sids = sids[:int(max_scenarios)]

    rows = []
    for sid in sids:
        res = simulate_soc(
            sid,
            scenario_df,
            sample_df,
            power_model=power_model,
            E_eff_wh=E_eff_wh,
            temp_model=temp_model,
            soh_model=soh_model,
            residual_power_w=residual_power_w,
            options=options,
        )
        if res.get("ok"):
            rows.append(res)
    return pd.DataFrame(rows)



def save_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(df) == 0:
        return out
    if "t_empty_hours" in df.columns:
        out["mean_t_empty_hours"] = float(df["t_empty_hours"].mean())
        out["median_t_empty_hours"] = float(df["t_empty_hours"].median())
    if "mae_soc" in df.columns:
        out["mean_mae_soc"] = float(df["mae_soc"].dropna().mean())
    if "rmse_soc" in df.columns:
        out["mean_rmse_soc"] = float(df["rmse_soc"].dropna().mean())
    return out


def plot_tornado(sens_df: pd.DataFrame, out_path: str, title: str = "OAT sensitivity on time-to-empty"):
    if len(sens_df) == 0:
        return
    df = sens_df.copy().sort_values("abs_delta_percent", ascending=True)
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(df["parameter"], df["delta_percent"])
    plt.xlabel("Δ time-to-empty (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def parse_args_common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=str, required=True, help="Path to data.xlsx (devices/sessions/samples)")
    p.add_argument("--outdir", type=str, default="task3_results", help="Output directory")
    p.add_argument("--max_scenarios", type=int, default=None, help="Limit scenarios for quick runs")
    p.add_argument("--extend_mode", type=str, default="last", choices=["last", "mean"], help="Extension mode when SOC doesn't reach 0")
    p.add_argument("--max_extend_hours", type=float, default=48.0, help="Max extension time horizon")
    return p
