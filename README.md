# Smartphone Battery Time-to-Empty (TTE) Prediction Model

This repository contains the implementation of a continuous-time model for predicting the **Time-to-Empty (TTE)** of smartphone batteries based on real-world usage and environmental conditions. The model integrates **equivalent circuit models (ECM)** and **energy balance equations** to estimate battery state-of-charge (SOC) and predict its remaining runtime.

## Overview

Battery life prediction is crucial for smartphone users, especially as devices grow in computational power and the complexity of their use cases increases. This project aims to provide a more accurate, scenario-specific prediction for TTE by utilizing:
- **Physics-based continuous-time model**
- **Real-world data from multiple smartphone brands** (including Apple, Huawei, Xiaomi)
- **Dynamic adjustments based on power consumption factors** like screen brightness, network usage, CPU load, and GPS

## Key Features

- **Real-time TTE Prediction**: Predict battery depletion time under various usage scenarios and conditions.
- **Usage Scenario Modeling**: Accounts for factors such as screen brightness, network activity, and GPS usage.
- **Temperature and Aging Effects**: Incorporates the impact of temperature and battery aging (State of Health, SOH).
- **Actionable Recommendations**: Provides user-friendly suggestions to optimize battery usage based on predictions.


