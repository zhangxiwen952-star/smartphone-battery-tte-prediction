import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import json
import os
from datetime import datetime
import seaborn as sns

plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

warnings.filterwarnings('ignore')

RESULTS_DIR = "battery_analysis_final"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class BatteryAnalysisModel:
    def __init__(self):
        self.power_model = None
        self.feature_columns = None
        self.scenario_data = None
        self.sample_data = None

        self.device_params = {
            'iPhone 14 Pro Max': {'E_full': 16.68, 'SOH': 95, 'V_nom': 3.86},
            'Xiaomi 14 Pro': {'E_full': 19.0, 'SOH': 95, 'V_nom': 3.87},
            'Huawei Mate 60 Pro+': {'E_full': 18.5, 'SOH': 95, 'V_nom': 3.82}
        }

    def load_data(self, file_path):
        try:
            self.scenario_data = pd.read_excel(file_path, sheet_name=1)
            self.sample_data = pd.read_excel(file_path, sheet_name=2)
            print(f"Loaded scenario {len(self.scenario_data)} rows")
            print(f"Loaded sample {len(self.sample_data)} rows")
            return True
        except Exception as e:
            print(f"Error loading {e}")
            return False

    def preprocess_data(self):
        if self.scenario_data is None or self.sample_data is None:
            return None, None

        df_scenario = self.scenario_data.copy()
        df_sample = self.sample_data.copy()

        if 't_秒' in df_sample.columns:
            df_sample['时间小时'] = df_sample['t_秒'] / 3600.0

        if all(col in df_sample.columns for col in ['上行KB_间隔', '下行KB_间隔']):
            df_sample['数据速率_KB/s'] = (df_sample['上行KB_间隔'] + df_sample['下行KB_间隔']) / 10.0
        else:
            df_sample['数据速率_KB/s'] = 0.0

        if '信号强度dBm' in df_sample.columns:
            df_sample['信号强度指标'] = -df_sample['信号强度dBm']
        else:
            df_sample['信号强度指标'] = 0.0

        network_mapping = {'无': 0, '5G': 1, '4G': 0.5, 'WiFi': 0.2}
        if '网络类型' in df_sample.columns:
            df_sample['网络类型编码'] = df_sample['网络类型'].map(network_mapping).fillna(0)
        else:
            df_sample['网络类型编码'] = 0

        if '定位服务' in df_sample.columns:
            df_sample['GPS开启'] = (df_sample['定位服务'] > 0).astype(int)
        else:
            df_sample['GPS开启'] = 0

        numeric_cols = ['SOC真实值%', '估计功耗W', '亮度%', 'CPU利用率%',
                        '数据速率_KB/s', '信号强度指标', '网络类型编码', 'GPS开启']

        for col in numeric_cols:
            if col in df_sample.columns:
                df_sample[col].fillna(df_sample[col].mean(), inplace=True)

        return df_scenario, df_sample

    def train_power_model(self, sample_df, features=None):
        if sample_df is None or '估计功耗W' not in sample_df.columns:
            return 0, 0

        if features is None:
            features = []
            potential_features = ['亮度%', 'CPU利用率%', '数据速率_KB/s',
                                  '信号强度指标', '网络类型编码', 'GPS开启']

            for feat in potential_features:
                if feat in sample_df.columns:
                    features.append(feat)

        if not features:
            return 0, 0

        X = sample_df[features]
        y = sample_df['估计功耗W']

        self.power_model = LinearRegression()
        self.power_model.fit(X, y)
        self.feature_columns = features

        y_pred = self.power_model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print("Power Model Training Complete")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} W")

        return r2, rmse

    def calculate_effective_capacity(self, scenario_df, sample_df):
        capacities = []

        for scenario_id in scenario_df['场景ID'].unique():
            scenario_info = scenario_df[scenario_df['场景ID'] == scenario_id].iloc[0]
            scenario_samples = sample_df[sample_df['场景ID'] == scenario_id]

            if len(scenario_samples) < 2:
                continue

            total_power = scenario_samples['估计功耗W'].sum() * (10 / 3600)

            soc_start = scenario_samples['SOC真实值%'].iloc[0]
            soc_end = scenario_samples['SOC真实值%'].iloc[-1]
            soc_change = soc_start - soc_end

            if soc_change > 0 and total_power > 0:
                E_eff = total_power / (soc_change / 100)
                capacities.append(E_eff)

        if capacities:
            avg_capacity = np.mean(capacities)
            print(f"Effective capacity calculated: {avg_capacity:.2f} Wh")
            return avg_capacity
        else:
            default_capacity = 16.0
            print(f"Using default capacity: {default_capacity} Wh")
            return default_capacity

    def predict_time_to_empty(self, scenario_df, sample_df, E_eff):
        time_results = {}

        for scenario_id in scenario_df['场景ID'].unique():
            scenario_info = scenario_df[scenario_df['场景ID'] == scenario_id].iloc[0]
            scenario_samples = sample_df[sample_df['场景ID'] == scenario_id]

            if len(scenario_samples) < 2:
                continue

            avg_power = scenario_samples['估计功耗W'].mean()
            time_to_empty = E_eff / avg_power

            hours = int(time_to_empty)
            minutes = int((time_to_empty - hours) * 60)

            time_results[scenario_id] = {
                'Scenario': scenario_info.get('场景名称', 'Unknown'),
                'Device': scenario_info.get('设备型号', 'Unknown'),
                'Avg Power (W)': avg_power,
                'Time to Empty (hours)': time_to_empty,
                'Formatted Time': f"{hours}h {minutes}min",
                'Brightness': scenario_info.get('亮度设定%', 50),
                'Network': scenario_info.get('网络类型设定', 'Unknown'),
                'GPS': scenario_info.get('定位服务(0/1)', 0),
                'Temperature': scenario_info.get('环境温度℃设定', 25)
            }

        print(f"Time-to-empty predictions completed for {len(time_results)} scenarios")
        return time_results

    def plot_soc_depletion_timeline(self, time_results, E_eff):
        if not time_results:
            return None

        scenarios_sorted = sorted(time_results.items(),
                                  key=lambda x: x[1]['Time to Empty (hours)'])

        if len(scenarios_sorted) < 3:
            return None

        short_scenario = scenarios_sorted[0]
        medium_scenario = scenarios_sorted[len(scenarios_sorted) // 2]
        long_scenario = scenarios_sorted[-1]

        selected_scenarios = {
            'Shortest Life': short_scenario,
            'Medium Life': medium_scenario,
            'Longest Life': long_scenario
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Battery SOC Depletion Timeline Analysis', fontsize=16, fontweight='bold')

        colors = ['#E63946', '#457B9D', '#2A9D8F']
        line_styles = ['-', '--', '-.']
        markers = ['o', 's', '^']

        time_range = np.linspace(0, 48, 500)

        for idx, (label, (scenario_id, result)) in enumerate(selected_scenarios.items()):
            tte = result['Time to Empty (hours)']
            power = result['Avg Power (W)']

            soc_values = np.maximum(0, 100 - (100 / tte) * time_range)

            ax1.plot(time_range, soc_values,
                     color=colors[idx], linestyle=line_styles[idx],
                     marker=markers[idx], markersize=4, markevery=20,
                     linewidth=2.5, alpha=0.9, label=f'{label}: {tte:.1f}h')

            ax1.axvline(x=tte, color=colors[idx], linestyle=':', alpha=0.7)
            ax1.text(tte + 0.3, 5, f'{tte:.1f}h',
                     color=colors[idx], fontsize=10, fontweight='bold')

        ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold')
        ax1.set_title('SOC Depletion from 100% to 0% for Different Usage Scenarios',
                      fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 48)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=11)

        ax1.axhline(y=20, color='red', linestyle='--', alpha=0.6, label='Low Battery (20%)')
        ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.6, label='Critical (10%)')
        ax1.axhline(y=5, color='darkred', linestyle='--', alpha=0.6, label='Emergency (5%)')

        ax2_bg = ax2.twinx()

        for idx, (label, (scenario_id, result)) in enumerate(selected_scenarios.items()):
            tte = result['Time to Empty (hours)']
            power = result['Avg Power (W)']
            energy_consumed = power * time_range

            ax2.axhline(y=power, color=colors[idx], linestyle=line_styles[idx],
                        linewidth=2, alpha=0.8, label=f'{label} Power: {power:.3f}W')

            ax2_bg.plot(time_range, energy_consumed, color=colors[idx],
                        linestyle=line_styles[idx], linewidth=2, alpha=0.4)

        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Power Consumption (W)', fontsize=12, fontweight='bold')
        ax2_bg.set_ylabel('Cumulative Energy (Wh)', fontsize=12, fontweight='bold')
        ax2.set_title('Power Consumption and Energy Accumulation', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 48)
        ax2.grid(True, alpha=0.3)

        lines1, labels1 = ax2.get_legend_handles_labels()
        ax2.legend(lines1, labels1, loc='upper left', fontsize=10)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"soc_depletion_timeline_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"SOC depletion timeline saved: {filename}")

        plt.show()
        return fig

    def plot_performance_comparison(self, time_results):
        if not time_results:
            return None

        network_groups = {}
        for scenario_id, result in time_results.items():
            network = result['Network']
            if network not in network_groups:
                network_groups[network] = []
            network_groups[network].append(result['Time to Empty (hours)'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        network_avg = {}
        network_std = {}
        for network, times in network_groups.items():
            network_avg[network] = np.mean(times)
            network_std[network] = np.std(times)

        networks = list(network_avg.keys())
        avg_times = [network_avg[net] for net in networks]
        std_times = [network_std[net] for net in networks]

        x_pos = np.arange(len(networks))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax1.bar(x_pos, avg_times, yerr=std_times, capsize=5,
                       color=colors[:len(networks)], alpha=0.8)

        ax1.set_xlabel('Network Type')
        ax1.set_ylabel('Time to Empty (hours)')
        ax1.set_title('Battery Life by Network Technology')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(networks, rotation=15)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, avg, std in zip(bars, avg_times, std_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f'{avg:.1f}±{std:.1f}h', ha='center', va='bottom', fontweight='bold')

        brightness_groups = {}
        for scenario_id, result in time_results.items():
            brightness = result['Brightness']
            if brightness <= 20:
                group = '0-20%'
            elif brightness <= 40:
                group = '21-40%'
            elif brightness <= 60:
                group = '41-60%'
            elif brightness <= 80:
                group = '61-80%'
            else:
                group = '81-100%'

            if group not in brightness_groups:
                brightness_groups[group] = []
            brightness_groups[group].append(result['Time to Empty (hours)'])

        brightness_avg = {}
        for group, times in brightness_groups.items():
            brightness_avg[group] = np.mean(times)

        brightness_order = ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%']
        brightnesses = [b for b in brightness_order if b in brightness_avg]
        avg_times_bright = [brightness_avg[b] for b in brightnesses]

        x_pos_b = np.arange(len(brightnesses))
        bars2 = ax2.bar(x_pos_b, avg_times_bright,
                        color=['#2E86AB', '#3A7CA5', '#4C6C9D', '#5C5C95', '#6C4C8D'],
                        alpha=0.8)

        ax2.set_xlabel('Brightness Level')
        ax2.set_ylabel('Time to Empty (hours)')
        ax2.set_title('Battery Life vs Screen Brightness')
        ax2.set_xticks(x_pos_b)
        ax2.set_xticklabels(brightnesses)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, avg in zip(bars2, avg_times_bright):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f'{avg:.1f}h', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"performance_comparison_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved: {filename}")

        plt.show()
        return fig

    def generate_analysis_report(self, time_results, E_eff):
        print("\n" + "=" * 70)
        print("BATTERY TIME-TO-EMPTY ANALYSIS REPORT")
        print("=" * 70)

        all_times = [result['Time to Empty (hours)'] for result in time_results.values()]
        avg_time = np.mean(all_times)
        min_time = min(all_times)
        max_time = max(all_times)
        std_time = np.std(all_times)

        print(f"\nOverall Statistics:")
        print(f"Average Time to Empty: {avg_time:.2f} ± {std_time:.2f} hours")
        print(f"Shortest Battery Life: {min_time:.2f} hours")
        print(f"Longest Battery Life: {max_time:.2f} hours")
        print(f"Range: {max_time - min_time:.2f} hours")
        print(f"Effective Capacity: {E_eff:.2f} Wh")

        worst_scenarios = sorted(time_results.items(),
                                 key=lambda x: x[1]['Time to Empty (hours)'])[:3]
        best_scenarios = sorted(time_results.items(),
                                key=lambda x: x[1]['Time to Empty (hours)'], reverse=True)[:3]

        print(f"\nFastest Drain Scenarios:")
        for scenario_id, result in worst_scenarios:
            print(f"  {scenario_id}: {result['Scenario']}")
            print(f"     Time: {result['Time to Empty (hours)']:.2f}h, Power: {result['Avg Power (W)']:.3f}W")

        print(f"\nSlowest Drain Scenarios:")
        for scenario_id, result in best_scenarios:
            print(f"  {scenario_id}: {result['Scenario']}")
            print(f"     Time: {result['Time to Empty (hours)']:.2f}h, Power: {result['Avg Power (W)']:.3f}W")

        flight_times = [result['Time to Empty (hours)'] for result in time_results.values()
                        if '无' in result['Network'] or '飞行' in result['Network']]
        fiveg_times = [result['Time to Empty (hours)'] for result in time_results.values()
                       if '5G' in result['Network']]

        if flight_times and fiveg_times:
            avg_flight = np.mean(flight_times)
            avg_5g = np.mean(fiveg_times)
            improvement = ((avg_flight - avg_5g) / avg_5g) * 100

            print(f"\nKey Findings:")
            print(f"5G to Flight Mode Improvement: {improvement:.1f}%")
            print(f"5G Average: {avg_5g:.2f}h, Flight Mode: {avg_flight:.2f}h")

        print(f"\nOptimization Recommendations:")
        print("1. Use WiFi over 5G when available")
        print("2. Reduce brightness to 30-50% for optimal balance")
        print("3. Disable GPS for non-location apps")
        print("4. Enable flight mode in no-service areas")
        print("5. Avoid extreme temperatures")

        return {
            'average_time': float(avg_time),
            'min_time': float(min_time),
            'max_time': float(max_time),
            'effective_capacity': float(E_eff)
        }


def main():
    print("=== Battery Time-to-Empty Analysis ===")

    model = BatteryAnalysisModel()

    file_path = r"C:\Users\Administrator\PycharmProjects\清欠台账\data.xlsx"

    if not model.load_data(file_path):
        return None

    scenario_df, sample_df = model.preprocess_data()

    if scenario_df is None or sample_df is None:
        return None

    r2, rmse = model.train_power_model(sample_df)

    if r2 == 0 and rmse == 0:
        simple_features = ['亮度%', 'CPU利用率%']
        r2, rmse = model.train_power_model(sample_df, simple_features)

    if r2 == 0 and rmse == 0:
        return None

    E_eff = model.calculate_effective_capacity(scenario_df, sample_df)

    time_results = model.predict_time_to_empty(scenario_df, sample_df, E_eff)

    if not time_results:
        return None

    fig1 = model.plot_soc_depletion_timeline(time_results, E_eff)
    fig2 = model.plot_performance_comparison(time_results)

    report = model.generate_analysis_report(time_results, E_eff)

    result_data = {
        'time_results': time_results,
        'effective_capacity': E_eff,
        'power_model_rmse': rmse,
        'report': report
    }

    result_file = os.path.join(RESULTS_DIR, 'analysis_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {result_file}")

    summary_file = os.path.join(RESULTS_DIR, 'analysis_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Battery Time-to-Empty Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Scenarios Analyzed: {len(time_results)}\n")
        f.write(f"Average Battery Life: {report['average_time']:.2f} hours\n")
        f.write(f"Range: {report['min_time']:.2f} to {report['max_time']:.2f} hours\n")
        f.write(f"Effective Capacity: {report['effective_capacity']:.2f} Wh\n")
        f.write(f"Power Model RMSE: {rmse:.4f} W\n")

    print(f"Summary saved to: {summary_file}")
    print("=== Analysis Complete ===")

    return result_data


if __name__ == "__main__":
    result = main()