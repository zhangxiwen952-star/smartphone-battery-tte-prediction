import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import json
import seaborn as sns
from matplotlib import rcParams
import matplotlib
import os
from datetime import datetime

warnings.filterwarnings('ignore')




matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


RESULTS_DIR = "battery_analysis_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class BatterySOCModel:
    """
    Smartphone Battery SOC Prediction Model
    """

    def __init__(self, device_params=None):
        """
        Initialize model parameters
        """
        if device_params is None:

            self.device_params = {
                'iPhone 14 Pro Max': {'E_full': 16.68, 'SOH': 95, 'V_nom': 3.86},
                'Xiaomi 14 Pro': {'E_full': 19.0, 'SOH': 95, 'V_nom': 3.87},
                'Huawei Mate 60 Pro+': {'E_full': 18.5, 'SOH': 95, 'V_nom': 3.82}
            }
        else:
            self.device_params = device_params

        self.power_model = None
        self.feature_columns = None
        self.imputer = None
        self.feature_stats = {}
        self.scenario_data = None
        self.sample_data = None


        self.device_name_mapping = {
            'iPhone 14 Pro Max': 'iPhone 14 Pro Max',
            '苹果 iPhone 14 Pro Max': 'iPhone 14 Pro Max',
            '小米 14 Pro': 'Xiaomi 14 Pro',
            '小米14 Pro': 'Xiaomi 14 Pro',
            '华为 Mate 60 Pro+': 'Huawei Mate 60 Pro+',
            '华为Mate 60 Pro+': 'Huawei Mate 60 Pro+',
            '华为 Mate 60 Pro': 'Huawei Mate 60 Pro+',
        }


        self.scenario_mapping = {
            '待机': 'Standby',
            '视频播放': 'Video Playback',
            '游戏': 'Gaming',
            '网页浏览': 'Web Browsing',
            '通话': 'Call',
            '5G上网': '5G Internet',
            '4G上网': '4G Internet',
            'WiFi上网': 'WiFi Internet',
            '导航': 'Navigation',
            '拍照': 'Photo Taking',
            '录像': 'Video Recording',
            '社交软件': 'Social Media',
            '音乐播放': 'Music Playback',
            '电子书': 'E-book Reading',
            '在线视频': 'Online Video',
            '下载': 'Downloading',
            '上传': 'Uploading',
            '混合场景': 'Mixed Scenario',
            '高负载': 'High Load',
            '中负载': 'Medium Load',
            '低负载': 'Low Load',
            '典型使用': 'Typical Usage',
            '极端测试': 'Extreme Test',
            '省电模式': 'Power Saving Mode',
            '性能模式': 'Performance Mode',
        }

    def load_data(self, file_path):
        """
        Load data from Excel file
        """
        try:

            print("Loading scenario configuration data...")
            self.scenario_data = pd.read_excel(file_path, sheet_name=1)


            print("Loading sampling data...")
            self.sample_data = pd.read_excel(file_path, sheet_name=2)

            print(f"Scenario data: {len(self.scenario_data)} rows")
            print(f"Sample data: {len(self.sample_data)} rows")

            return True
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    def translate_device_name(self, device_name):
        """Translate device name to English"""
        if device_name in self.device_name_mapping:
            return self.device_name_mapping[device_name]
        return device_name

    def translate_scenario_name(self, scenario_name):
        """Translate scenario name to English"""

        result = scenario_name
        for chinese, english in self.scenario_mapping.items():
            if chinese in result:
                result = result.replace(chinese, english)
        return result

    def preprocess_scenario_data(self):
        """
        Preprocess scenario configuration data
        """
        if self.scenario_data is None:
            return None

        df = self.scenario_data.copy()


        if '设备型号' in df.columns:
            df['Device Model'] = df['设备型号'].apply(self.translate_device_name)


        if '场景名称' in df.columns:
            df['Scenario Name'] = df['场景名称'].apply(self.translate_scenario_name)



        if '网络类型设定' in df.columns:
            df['Network Type Code'] = df['网络类型设定'].map({
                '无/飞行模式': 0,
                '5G': 1,
                '4G': 0.5,
                'WiFi': 0.2
            }).fillna(0)
        else:
            df['Network Type Code'] = 0


        if 'GPS模式' in df.columns:
            df['GPS Code'] = df['GPS模式'].map({
                '关闭': 0,
                '高精度': 1
            }).fillna(0)
        else:
            df['GPS Code'] = 0


        if '飞行模式(0/1)' in df.columns:
            df['Flight Mode Code'] = df['飞行模式(0/1)']


        if '定位服务(0/1)' in df.columns:
            df['Location Service Code'] = df['定位服务(0/1)']


        if all(col in df.columns for col in ['起始SOC%', '结束SOC%']):
            df['SOC Change'] = df['起始SOC%'] - df['结束SOC%']

        print("Scenario data preprocessing completed")
        return df

    def preprocess_sample_data(self):
        """
        Preprocess sampling data
        """
        if self.sample_data is None:
            return None

        df = self.sample_data.copy()



        df['Time (hours)'] = df['t_秒'] / 3600.0


        if all(col in df.columns for col in ['上行KB_间隔', '下行KB_间隔']):
            df['Data Rate (KB/s)'] = (df['上行KB_间隔'] + df['下行KB_间隔']) / 10.0
        else:
            df['Data Rate (KB/s)'] = 0.0


        if '信号强度dBm' in df.columns:
            df['Signal Strength Index'] = -df['信号强度dBm']
        else:
            df['Signal Strength Index'] = 0.0


        network_mapping = {'无': 0, '5G': 1, '4G': 0.5, 'WiFi': 0.2}
        if '网络类型' in df.columns:
            df['Network Type Code'] = df['网络类型'].map(network_mapping).fillna(0)
        else:
            df['Network Type Code'] = 0


        if '定位服务' in df.columns:
            df['GPS On'] = (df['定位服务'] > 0).astype(int)
        else:
            df['GPS On'] = 0


        numeric_cols = ['SOC真实值%', '估计功耗W', '亮度%', 'CPU利用率%',
                        'Data Rate (KB/s)', 'Signal Strength Index',
                        'Network Type Code', 'GPS On']

        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)

        print("Sample data preprocessing completed")
        return df

    def train_power_model(self, df, features=None):
        """
        Train power consumption prediction model
        """
        if df is None or '估计功耗W' not in df.columns:
            print("Error: No power consumption data")
            return 0, 0


        if features is None:
            features = []
            potential_features = ['亮度%', 'CPU利用率%', 'Data Rate (KB/s)',
                                  'Signal Strength Index', 'Network Type Code', 'GPS On']

            for feat in potential_features:
                if feat in df.columns:
                    features.append(feat)

        if not features:
            print("Error: No available features")
            return 0, 0

        print(f"Using features: {features}")


        X = df[features]
        y = df['估计功耗W']


        self.power_model = LinearRegression()
        self.power_model.fit(X, y)
        self.feature_columns = features


        y_pred = self.power_model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print("\n" + "=" * 50)
        print("Power Model Training Results:")
        print(f"Training samples: {len(X)}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} W")
        print("\nModel Coefficients:")
        for feat, coef in zip(features, self.power_model.coef_):
            print(f"  {feat}: {coef:.6f}")
        print(f"Intercept: {self.power_model.intercept_:.6f}")
        print("=" * 50)

        return r2, rmse

    def predict_power(self, features_dict):
        """
        Predict power consumption
        """
        if self.power_model is None or self.feature_columns is None:
            return 0.0


        features = []
        for feat in self.feature_columns:
            if feat in features_dict:
                features.append(features_dict[feat])
            else:
                features.append(0.0)


        return self.power_model.predict([features])[0]

    def calculate_effective_capacity(self, scenario_df, sample_df):
        """
        Calculate effective battery capacity
        """

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

                print(
                    f"Scenario {scenario_id}: SOC change {soc_change:.2f}%, Total power {total_power:.4f}Wh, Estimated capacity {E_eff:.2f}Wh")

        if capacities:
            avg_capacity = np.mean(capacities)
            print(f"\nAverage effective capacity: {avg_capacity:.2f}Wh")
            return avg_capacity
        else:

            default_capacity = 16.0
            print(f"Cannot calculate effective capacity, using default: {default_capacity}Wh")
            return default_capacity

    def simulate_soc_for_scenario(self, scenario_id, scenario_df, sample_df, E_eff):
        """
        Simulate SOC changes for a specific scenario
        """

        scenario_samples = sample_df[sample_df['场景ID'] == scenario_id].copy()

        if len(scenario_samples) < 2:
            print(f"Insufficient data for scenario {scenario_id}")
            return None, None


        scenario_samples = scenario_samples.sort_values('t_秒')


        time_values = scenario_samples['t_秒'].values / 3600.0
        soc_values = [scenario_samples['SOC真实值%'].iloc[0]]


        for i in range(1, len(scenario_samples)):

            dt = (scenario_samples['t_秒'].iloc[i] - scenario_samples['t_秒'].iloc[i - 1]) / 3600.0


            features = {}
            for feat in self.feature_columns:
                if feat in scenario_samples.columns:
                    features[feat] = scenario_samples.iloc[i][feat]
                else:
                    features[feat] = 0.0


            power = self.predict_power(features)


            dsoc = -power / E_eff * 100 * dt


            new_soc = soc_values[-1] + dsoc
            soc_values.append(max(0, min(new_soc, 100)))

        return time_values, np.array(soc_values)

    def analyze_scenario_performance(self, scenario_df, sample_df, E_eff):
        """
        Analyze performance for each scenario
        """
        results = {}

        for scenario_id in scenario_df['场景ID'].unique():
            scenario_info = scenario_df[scenario_df['场景ID'] == scenario_id].iloc[0]
            scenario_samples = sample_df[sample_df['场景ID'] == scenario_id]

            if len(scenario_samples) < 2:
                continue


            t_pred, soc_pred = self.simulate_soc_for_scenario(scenario_id, scenario_df, sample_df, E_eff)

            if t_pred is None:
                continue


            soc_actual = scenario_samples.sort_values('t_秒')['SOC真实值%'].values
            soc_pred_aligned = np.interp(scenario_samples['t_秒'], t_pred * 3600, soc_pred)

            mae = np.mean(np.abs(soc_actual - soc_pred_aligned))
            rmse = np.sqrt(mean_squared_error(soc_actual, soc_pred_aligned))


            device_name = scenario_info.get('Device Model', scenario_info.get('设备型号', 'Unknown'))


            scenario_name = scenario_info.get('Scenario Name', scenario_info.get('场景名称', 'Unknown'))

            results[scenario_id] = {
                'Device': device_name,
                'Scenario': scenario_name,
                'Brightness': scenario_info.get('亮度设定%', scenario_info.get('亮度%', 0)),
                'Network': scenario_info.get('网络类型设定', 'Unknown'),
                'GPS': scenario_info.get('定位服务(0/1)', 0),
                'Temperature': scenario_info.get('环境温度℃设定', 25),
                'MAE': mae,
                'RMSE': rmse,
                'Actual SOC Change': scenario_info.get('起始SOC%', 100) - scenario_info.get('结束SOC%', 0),
                'Predicted SOC Change': soc_pred[0] - soc_pred[-1] if len(soc_pred) > 0 else 0
            }

            print(f"Scenario {scenario_id}: MAE={mae:.4f}%, RMSE={rmse:.4f}%")

        return results

    def plot_scenario_comparison(self, scenario_df, sample_df, E_eff, max_plots=8):
        """
        Plot scenario comparison
        """

        scenarios_to_plot = scenario_df['场景ID'].unique()[:max_plots]
        n_scenarios = len(scenarios_to_plot)

        if n_scenarios == 0:
            return


        n_cols = min(3, n_scenarios)
        n_rows = (n_scenarios + n_cols - 1) // n_cols


        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows))
        fig.suptitle('SOC Prediction Comparison for Different Scenarios',
                     fontsize=18, fontweight='bold', y=1.02)

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()


        actual_color = '#2E86AB'  # Blue
        pred_color = '#E63946'  # Red
        fill_color = '#F4A261'  # Orange for fill
        grid_color = '#E0E0E0'

        for idx, scenario_id in enumerate(scenarios_to_plot):
            ax = axes[idx]
            scenario_info = scenario_df[scenario_df['场景ID'] == scenario_id].iloc[0]
            scenario_samples = sample_df[sample_df['场景ID'] == scenario_id]

            if len(scenario_samples) < 2:
                continue


            t_pred, soc_pred = self.simulate_soc_for_scenario(scenario_id, scenario_df, sample_df, E_eff)

            if t_pred is None:
                continue


            scenario_samples = scenario_samples.sort_values('t_秒')


            ax.plot(scenario_samples['t_秒'] / 3600, scenario_samples['SOC真实值%'],
                    color=actual_color, linewidth=2.5, label='Actual SOC',
                    marker='o', markersize=4, markevery=8, alpha=0.9)


            ax.plot(t_pred, soc_pred, color=pred_color, linewidth=2,
                    label='Predicted SOC', linestyle='--', marker='s',
                    markersize=4, markevery=8, alpha=0.9)


            soc_actual_interp = np.interp(t_pred, scenario_samples['t_秒'] / 3600,
                                          scenario_samples['SOC真实值%'])
            ax.fill_between(t_pred, soc_actual_interp, soc_pred,
                            alpha=0.15, color=fill_color, label='Prediction Error')


            device_name = scenario_info.get('Device Model', scenario_info.get('设备型号', 'Unknown Device'))
            scenario_name = scenario_info.get('Scenario Name', scenario_info.get('场景名称', 'Unknown Scenario'))
            title = f"{device_name}\n{scenario_name}"

            ax.set_title(title, fontsize=12, fontweight='semibold', pad=12)
            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='medium')
            ax.set_ylabel('SOC (%)', fontsize=11, fontweight='medium')


            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5, color=grid_color)


            y_min = max(0, min(scenario_samples['SOC真实值%'].min(), soc_pred.min()) - 2)
            y_max = min(100, max(scenario_samples['SOC真实值%'].max(), soc_pred.max()) + 2)
            ax.set_ylim([y_min, y_max])


            soc_pred_aligned = np.interp(scenario_samples['t_秒'], t_pred * 3600, soc_pred)
            mae = np.mean(np.abs(scenario_samples['SOC真实值%'] - soc_pred_aligned))


            textstr = f'MAE: {mae:.3f}%\nScenario ID: {scenario_id}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                         edgecolor='gray', linewidth=0.5)
            ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=props)


            ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                      edgecolor='gray', fancybox=True)


            ax.tick_params(axis='both', which='major', labelsize=9)


        for idx in range(n_scenarios, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"scenario_comparison_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Scenario comparison plot saved as: {filename}")

        plt.show()

    def plot_feature_analysis(self, sample_df):
        """
        Plot feature analysis
        """
        if self.power_model is None or self.feature_columns is None:
            return


        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle('Feature Analysis and Power Consumption Relationships',
                     fontsize=18, fontweight='bold', y=1.02)


        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        grid_color = '#E0E0E0'


        ax1 = axes[0, 0]
        if '亮度%' in sample_df.columns and '估计功耗W' in sample_df.columns:

            scatter = ax1.scatter(sample_df['亮度%'], sample_df['估计功耗W'],
                                  c=sample_df['亮度%'], cmap='viridis',
                                  alpha=0.6, s=15, marker='o', edgecolor='black', linewidth=0.5)

            ax1.set_xlabel('Brightness (%)', fontsize=12, fontweight='medium')
            ax1.set_ylabel('Power Consumption (W)', fontsize=12, fontweight='medium')
            ax1.set_title('Brightness vs Power Consumption', fontsize=13, fontweight='semibold')
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color)


            cb = fig.colorbar(scatter, ax=ax1)
            cb.set_label('Brightness (%)', fontsize=10)


            z = np.polyfit(sample_df['亮度%'], sample_df['估计功耗W'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(sample_df['亮度%'].min(), sample_df['亮度%'].max(), 100)
            ax1.plot(x_range, p(x_range), "r-", alpha=0.8, linewidth=2,
                     label=f'Linear Fit: y={z[0]:.4f}x+{z[1]:.4f}')
            ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)


            ax1.tick_params(axis='both', which='major', labelsize=9)


        ax2 = axes[0, 1]
        if 'CPU利用率%' in sample_df.columns and '估计功耗W' in sample_df.columns:

            scatter = ax2.scatter(sample_df['CPU利用率%'], sample_df['估计功耗W'],
                                  c=sample_df['CPU利用率%'], cmap='plasma',
                                  alpha=0.6, s=15, marker='s', edgecolor='black', linewidth=0.5)

            ax2.set_xlabel('CPU Usage (%)', fontsize=12, fontweight='medium')
            ax2.set_ylabel('Power Consumption (W)', fontsize=12, fontweight='medium')
            ax2.set_title('CPU Usage vs Power Consumption', fontsize=13, fontweight='semibold')
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color)


            cb2 = fig.colorbar(scatter, ax=ax2)
            cb2.set_label('CPU Usage (%)', fontsize=10)


            z2 = np.polyfit(sample_df['CPU利用率%'], sample_df['估计功耗W'], 1)
            p2 = np.poly1d(z2)
            x_range2 = np.linspace(sample_df['CPU利用率%'].min(),
                                   sample_df['CPU利用率%'].max(), 100)
            ax2.plot(x_range2, p2(x_range2), "r-", alpha=0.8, linewidth=2,
                     label=f'Linear Fit: y={z2[0]:.4f}x+{z2[1]:.4f}')
            ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)


            ax2.tick_params(axis='both', which='major', labelsize=9)


        ax3 = axes[1, 0]
        if '网络类型' in sample_df.columns and '估计功耗W' in sample_df.columns:
            network_power = sample_df.groupby('网络类型')['估计功耗W'].agg(['mean', 'std']).fillna(0)
            networks = network_power.index.tolist()


            network_labels = {
                '无': 'None/Offline',
                '5G': '5G',
                '4G': '4G',
                'WiFi': 'WiFi',
                '无/飞行模式': 'Flight Mode',
                '飞行模式': 'Flight Mode'
            }
            english_labels = [network_labels.get(net, net) for net in networks]

            x_pos = np.arange(len(networks))
            bars = ax3.bar(x_pos, network_power['mean'], yerr=network_power['std'],
                           capsize=5, alpha=0.8, color=colors[:len(networks)],
                           edgecolor='black', linewidth=1)

            ax3.set_xlabel('Network Type', fontsize=12, fontweight='medium')
            ax3.set_ylabel('Average Power Consumption (W)', fontsize=12, fontweight='medium')
            ax3.set_title('Average Power Consumption by Network Type',
                          fontsize=13, fontweight='semibold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(english_labels, fontsize=10, rotation=0)
            ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color, axis='y')


            for i, (bar, mean_val) in enumerate(zip(bars, network_power['mean'])):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                         f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='medium')


            ax3.tick_params(axis='both', which='major', labelsize=9)


        ax4 = axes[1, 1]
        if self.power_model is not None and self.feature_columns is not None:
            coefficients = self.power_model.coef_


            feature_names = []
            for feat in self.feature_columns:
                if feat == '亮度%':
                    feature_names.append('Brightness (%)')
                elif feat == 'CPU利用率%':
                    feature_names.append('CPU Usage (%)')
                elif feat == 'Data Rate (KB/s)':
                    feature_names.append('Data Rate (KB/s)')
                elif feat == 'Signal Strength Index':
                    feature_names.append('Signal Strength')
                elif feat == 'Network Type Code':
                    feature_names.append('Network Type')
                elif feat == 'GPS On':
                    feature_names.append('GPS Status')
                else:
                    feature_names.append(feat)

            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).sort_values('Coefficient', key=abs, ascending=True)


            y_pos = np.arange(len(feature_importance))
            colors_bar = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(feature_importance)))
            bars = ax4.barh(y_pos, feature_importance['Coefficient'],
                            align='center', alpha=0.8,
                            color=colors_bar,
                            edgecolor='black', linewidth=1)

            ax4.set_xlabel('Coefficient Value', fontsize=12, fontweight='medium')
            ax4.set_title('Feature Importance in Power Consumption Model',
                          fontsize=13, fontweight='semibold')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(feature_importance['Feature'], fontsize=10)
            ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color, axis='x')


            ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)


            for i, (bar, coeff) in enumerate(zip(bars, feature_importance['Coefficient'])):
                width = bar.get_width()
                ha = 'left' if width >= 0 else 'right'
                x_pos = width + (0.001 if width >= 0 else -0.001)
                ax4.text(x_pos, bar.get_y() + bar.get_height() / 2.,
                         f'{coeff:.4f}', ha=ha, va='center', fontsize=9, fontweight='medium')


            ax4.tick_params(axis='both', which='major', labelsize=9)

        plt.tight_layout()


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"feature_analysis_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Feature analysis plot saved as: {filename}")

        plt.show()

    def generate_report(self, scenario_results, E_eff):
        """
        Generate analysis report
        """
        print("\n" + "=" * 60)
        print("Battery SOC Prediction Model Analysis Report")
        print("=" * 60)


        all_mae = [result['MAE'] for result in scenario_results.values()]
        all_rmse = [result['RMSE'] for result in scenario_results.values()]

        print(f"\nOverall Performance Metrics:")
        print(f"Average MAE: {np.mean(all_mae):.4f}%")
        print(f"Average RMSE: {np.mean(all_rmse):.4f}%")
        print(f"Maximum MAE: {np.max(all_mae):.4f}%")
        print(f"Minimum MAE: {np.min(all_mae):.4f}%")
        print(f"Estimated Effective Capacity: {E_eff:.2f} Wh")


        print(f"\nPerformance by Device:")
        devices = set(result['Device'] for result in scenario_results.values())
        for device in devices:
            device_results = [r for r in scenario_results.values() if r['Device'] == device]
            device_mae = np.mean([r['MAE'] for r in device_results])
            print(f"  {device}: Average MAE = {device_mae:.4f}%")


        print(f"\nPerformance by Scenario Type:")
        scenario_types = set(result['Scenario'] for result in scenario_results.values())
        for scenario_type in scenario_types:
            type_results = [r for r in scenario_results.values() if r['Scenario'] == scenario_type]
            if type_results:
                type_mae = np.mean([r['MAE'] for r in type_results])
                print(f"  {scenario_type}: Average MAE = {type_mae:.4f}%")


        print(f"\nPerformance by Brightness Level:")
        brightness_levels = sorted(set(result['Brightness'] for result in scenario_results.values()))
        for brightness in brightness_levels:
            bright_results = [r for r in scenario_results.values() if r['Brightness'] == brightness]
            if bright_results:
                bright_mae = np.mean([r['MAE'] for r in bright_results])
                print(f"  Brightness {brightness}%: Average MAE = {bright_mae:.4f}%")


        print(f"\nPerformance by Network Type:")
        network_types = set(result['Network'] for result in scenario_results.values())
        for network in network_types:
            net_results = [r for r in scenario_results.values() if r['Network'] == network]
            if net_results:
                net_mae = np.mean([r['MAE'] for r in net_results])
                print(f"  {network}: Average MAE = {net_mae:.4f}%")


        print(f"\nPerformance by Environment Temperature:")
        temps = sorted(set(result['Temperature'] for result in scenario_results.values()))
        for temp in temps:
            temp_results = [r for r in scenario_results.values() if r['Temperature'] == temp]
            if temp_results:
                temp_mae = np.mean([r['MAE'] for r in temp_results])
                print(f"  {temp}°C: Average MAE = {temp_mae:.4f}%")


        print(f"\nDetailed Scenario Results:")
        print(
            f"{'Scenario ID':<12} {'Device':<20} {'Scenario':<25} {'Bright':<8} {'Network':<12} {'GPS':<6} {'Temp':<6} {'MAE':<10} {'RMSE':<10}")
        print("-" * 110)

        for scenario_id, result in scenario_results.items():
            device_short = result['Device'][:18] + '...' if len(result['Device']) > 18 else result['Device']
            scenario_short = result['Scenario'][:22] + '...' if len(result['Scenario']) > 22 else result['Scenario']
            network_short = result['Network'][:10] if len(result['Network']) <= 10 else result['Network'][:7] + '...'

            print(f"{scenario_id:<12} {device_short:<20} {scenario_short:<25} "
                  f"{result['Brightness']:<8} {network_short:<12} {result['GPS']:<6} "
                  f"{result['Temperature']:<6} {result['MAE']:.4f}% {result['RMSE']:.4f}%")

        print("=" * 60)

        return {
            'Average MAE': float(np.mean(all_mae)),
            'Average RMSE': float(np.mean(all_rmse)),
            'Effective Capacity': float(E_eff),
            'Device Performance': {
                device: float(np.mean([r['MAE'] for r in scenario_results.values() if r['Device'] == device]))
                for device in devices},
            'Scenario Performance': {scenario_type: float(
                np.mean([r['MAE'] for r in scenario_results.values() if r['Scenario'] == scenario_type]))
                                     for scenario_type in scenario_types}
        }


def generate_recommendations(report, scenario_results):
    """
    Generate optimization recommendations based on analysis
    """
    print("\n" + "=" * 60)
    print("Optimization Recommendations")
    print("=" * 60)


    worst_scenarios = sorted(scenario_results.items(),
                             key=lambda x: x[1]['MAE'], reverse=True)[:3]

    print("\nTop 3 Worst Performing Scenarios (Highest MAE):")
    for scenario_id, result in worst_scenarios:
        print(f"  {scenario_id}: {result['Scenario']} (MAE: {result['MAE']:.4f}%)")


    best_scenarios = sorted(scenario_results.items(),
                            key=lambda x: x[1]['MAE'])[:3]

    print("\nTop 3 Best Performing Scenarios (Lowest MAE):")
    for scenario_id, result in best_scenarios:
        print(f"  {scenario_id}: {result['Scenario']} (MAE: {result['MAE']:.4f}%)")


    device_performance = report.get('Device Performance', {})
    if device_performance:
        best_device = min(device_performance.items(), key=lambda x: x[1])
        worst_device = max(device_performance.items(), key=lambda x: x[1])

        print(f"\nDevice Performance Comparison:")
        print(f"  Best Device: {best_device[0]} (MAE: {best_device[1]:.4f}%)")
        print(f"  Worst Device: {worst_device[0]} (MAE: {worst_device[1]:.4f}%)")


    scenario_type_performance = report.get('Scenario Performance', {})
    if scenario_type_performance:
        best_scenario_type = min(scenario_type_performance.items(), key=lambda x: x[1])
        worst_scenario_type = max(scenario_type_performance.items(), key=lambda x: x[1])

        print(f"\nScenario Type Performance Comparison:")
        print(f"  Best Scenario Type: {best_scenario_type[0]} (MAE: {best_scenario_type[1]:.4f}%)")
        print(f"  Worst Scenario Type: {worst_scenario_type[0]} (MAE: {worst_scenario_type[1]:.4f}%)")


    avg_mae = report.get('Average MAE', 0)
    if avg_mae < 0.5:
        print(f"\nOverall Assessment: Excellent Model Performance (Average MAE: {avg_mae:.4f}%)")
        print("Recommendation: Model is suitable for practical applications")
    elif avg_mae < 1.0:
        print(f"\nOverall Assessment: Good Model Performance (Average MAE: {avg_mae:.4f}%)")
        print("Recommendation: Model is acceptable, consider further optimization")
    else:
        print(f"\nOverall Assessment: Model Performance Needs Improvement (Average MAE: {avg_mae:.4f}%)")
        print("Recommendation: Review model and feature selection")


    print(f"\nSpecific Optimization Suggestions:")
    print("1. Add More Features: Consider battery temperature, environmental conditions, signal quality")
    print("2. Improve Power Model: Try nonlinear models or ensemble learning methods")
    print("3. Enhance Data Quality: Ensure data collection accuracy and consistency")
    print("4. Scenario Segmentation: Train specialized models for different scenario types")
    print("5. Real-time Calibration: Calibrate model based on real-time usage data")
    print("6. Feature Engineering: Create interaction terms and higher-order features")
    print("7. Model Ensemble: Combine multiple models for better prediction accuracy")

    print("=" * 60)


def plot_performance_comparison(scenario_results):
    """
    Plot performance comparison charts
    """

    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    fig1.suptitle('Model Performance Comparison Across Devices and Scenarios',
                  fontsize=18, fontweight='bold', y=1.02)


    device_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    scenario_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9edae5']
    grid_color = '#E0E0E0'


    device_mae = {}
    for scenario_id, result in scenario_results.items():
        device = result['Device']
        if device not in device_mae:
            device_mae[device] = []
        device_mae[device].append(result['MAE'])


    device_avg_mae = {device: np.mean(mae_list) for device, mae_list in device_mae.items()}


    ax1 = axes1[0]
    devices = list(device_avg_mae.keys())
    mae_values = [device_avg_mae[device] for device in devices]


    x_pos = np.arange(len(devices))
    colors_bars = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(devices)))
    bars = ax1.bar(x_pos, mae_values, color=colors_bars,
                   alpha=0.9, edgecolor='black', linewidth=1.2,
                   width=0.65)

    ax1.set_xlabel('Device Model', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Average MAE (%)', fontsize=12, fontweight='medium')
    ax1.set_title('Performance Comparison by Device Model', fontsize=13, fontweight='semibold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(devices, rotation=15, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color, axis='y')


    for bar, value, device in zip(bars, mae_values, devices):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='medium')


    avg_all = np.mean(mae_values)
    ax1.axhline(y=avg_all, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Overall Average: {avg_all:.4f}%')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)


    ax1.tick_params(axis='both', which='major', labelsize=9)


    scenario_type_mae = {}
    for scenario_id, result in scenario_results.items():
        scenario_type = result['Scenario']
        if scenario_type not in scenario_type_mae:
            scenario_type_mae[scenario_type] = []
        scenario_type_mae[scenario_type].append(result['MAE'])


    scenario_avg_mae = {scenario_type: np.mean(mae_list)
                        for scenario_type, mae_list in scenario_type_mae.items()}


    ax2 = axes1[1]
    scenario_types = list(scenario_avg_mae.keys())
    mae_values_scenario = [scenario_avg_mae[scenario_type] for scenario_type in scenario_types]


    y_pos = np.arange(len(scenario_types))
    colors_bars2 = plt.cm.viridis(np.linspace(0.2, 0.8, len(scenario_types)))
    bars2 = ax2.barh(y_pos, mae_values_scenario, color=colors_bars2,
                     alpha=0.9, edgecolor='black', linewidth=1.2,
                     height=0.65)

    ax2.set_xlabel('Average MAE (%)', fontsize=12, fontweight='medium')
    ax2.set_ylabel('Scenario Type', fontsize=12, fontweight='medium')
    ax2.set_title('Performance Comparison by Scenario Type', fontsize=13, fontweight='semibold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(scenario_types, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color, axis='x')


    for bar, value, scenario in zip(bars2, mae_values_scenario, scenario_types):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height() / 2.,
                 f'{value:.4f}', ha='left', va='center', fontsize=9, fontweight='medium')


    avg_all_scenario = np.mean(mae_values_scenario)
    ax2.axvline(x=avg_all_scenario, color='red', linestyle='--', alpha=0.7,
                linewidth=1.5, label=f'Overall Average: {avg_all_scenario:.4f}%')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)


    ax2.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = os.path.join(RESULTS_DIR, f"performance_comparison_{timestamp}.png")
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"Performance comparison plot saved as: {filename1}")

    plt.show()


    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 7))

    all_mae_values = [result['MAE'] for result in scenario_results.values()]


    parts = ax3.violinplot(all_mae_values, showmeans=True, showmedians=True, showextrema=True)


    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)


    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('green')
    parts['cmedians'].set_linewidth(2)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(1)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(1)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(1)

    ax3.set_xlabel('MAE Distribution', fontsize=12, fontweight='medium')
    ax3.set_ylabel('Mean Absolute Error (%)', fontsize=12, fontweight='medium')
    ax3.set_title('MAE Distribution Across All Test Scenarios', fontsize=13, fontweight='semibold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color, axis='y')


    stats_text = (f'Mean: {np.mean(all_mae_values):.4f}%\n'
                  f'Median: {np.median(all_mae_values):.4f}%\n'
                  f'Std Dev: {np.std(all_mae_values):.4f}%\n'
                  f'Min: {np.min(all_mae_values):.4f}%\n'
                  f'Max: {np.max(all_mae_values):.4f}%\n'
                  f'IQR: {np.percentile(all_mae_values, 75) - np.percentile(all_mae_values, 25):.4f}%')

    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor='gray', linewidth=0.5)
    ax3.text(1.2, np.median(all_mae_values), stats_text, fontsize=9,
             va='center', bbox=props)


    ax3.set_xlim([0.7, 1.5])
    ax3.set_xticks([1])
    ax3.set_xticklabels(['MAE Distribution'], fontsize=10)
    ax3.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()


    filename2 = os.path.join(RESULTS_DIR, f"mae_distribution_{timestamp}.png")
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"MAE distribution plot saved as: {filename2}")

    plt.show()


def main():
    """
    Main function
    """
    print("=== Smartphone Battery SOC Prediction Model ===")
    print(f"Results will be saved in: {os.path.abspath(RESULTS_DIR)}")


    model = BatterySOCModel()


    file_path = r"data-cn.xlsx"


    print(f"\n1. Loading data: {file_path}")
    if not model.load_data(file_path):
        print("Failed to load data, exiting...")
        return None


    print("\n2. Data preprocessing")
    scenario_df = model.preprocess_scenario_data()
    sample_df = model.preprocess_sample_data()

    if scenario_df is None or sample_df is None:
        print("Data preprocessing failed, exiting...")
        return None


    print("\n3. Training power consumption model")
    r2, rmse = model.train_power_model(sample_df)

    if r2 == 0 and rmse == 0:
        print("Model training failed, trying simplified features...")

        simple_features = ['亮度%', 'CPU利用率%']
        r2, rmse = model.train_power_model(sample_df, simple_features)

    if r2 == 0 and rmse == 0:
        print("Power model training failed, cannot continue analysis")
        return None


    print("\n4. Calculating effective battery capacity")
    E_eff = model.calculate_effective_capacity(scenario_df, sample_df)


    print("\n5. Analyzing scenario performance")
    scenario_results = model.analyze_scenario_performance(scenario_df, sample_df, E_eff)

    if not scenario_results:
        print("No valid scenario results")
        return None


    print("\n6. Generating analysis report")
    report = model.generate_report(scenario_results, E_eff)


    print("\n7. Generating analysis charts")


    model.plot_scenario_comparison(scenario_df, sample_df, E_eff, max_plots=6)


    model.plot_feature_analysis(sample_df)


    plot_performance_comparison(scenario_results)


    print("\n8. Optimization recommendations")
    generate_recommendations(report, scenario_results)

    print("\n=== Analysis Completed ===")
    print(f"All results and plots saved in: {os.path.abspath(RESULTS_DIR)}")

    return report


if __name__ == "__main__":

    result = main()


    if result is not None:
        try:
            result_file = os.path.join(RESULTS_DIR, 'battery_analysis_result.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nAnalysis results saved to: {result_file}")


            summary_file = os.path.join(RESULTS_DIR, 'analysis_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("Battery SOC Model Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Average MAE: {result.get('Average MAE', 0):.4f}%\n")
                f.write(f"Average RMSE: {result.get('Average RMSE', 0):.4f}%\n")
                f.write(f"Effective Capacity: {result.get('Effective Capacity', 0):.2f} Wh\n\n")

                f.write("Device Performance:\n")
                for device, perf in result.get('Device Performance', {}).items():
                    f.write(f"  {device}: {perf:.4f}% MAE\n")

                f.write("\nScenario Performance:\n")
                for scenario, perf in result.get('Scenario Performance', {}).items():
                    f.write(f"  {scenario}: {perf:.4f}% MAE\n")

            print(f"Analysis summary saved to: {summary_file}")

        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("\nAnalysis failed, no results generated")