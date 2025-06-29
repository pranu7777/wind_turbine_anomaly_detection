import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def forecast_all_sensors(df, horizon=50, backtest_windows=3, output_dir='sensor_forecasts'):
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Clean and sort synthetic or existing timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    results_summary = []

    for sensor in sensor_cols:
        print(f"\n📈 Forecasting {sensor} using SARIMAX...")

        try:
            series_df = df[['timestamp', sensor]].dropna().copy()
            series_df = series_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
            series_df.set_index('timestamp', inplace=True)
            series_df.index = pd.DatetimeIndex(series_df.index)

            # Ensure regular frequency
            inferred_freq = pd.infer_freq(series_df.index)
            if inferred_freq:
                series_df.index.freq = inferred_freq
            else:
                series_df = series_df.asfreq('10S')  # fallback

            series = series_df[sensor]

            rmse_list = []
            forecasts = []

            for i in range(backtest_windows):
                split_point = int(len(series) * (0.7 - i * 0.1))
                train = series[:split_point]
                test = series[split_point:split_point + horizon]

                if len(test) < horizon:
                    print(f"⚠️ Not enough data in window {i} for {sensor}")
                    continue

                model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=horizon)

                rmse = mean_squared_error(test[:horizon], forecast, squared=False)
                rmse_list.append(rmse)

                forecast_df = pd.DataFrame({
                    'timestamp': test.index[:horizon],
                    'actual': test[:horizon].values,
                    'forecast': forecast.values
                })

                forecasts.append(forecast_df)

            if forecasts:
                # ✅ Select the forecast with the latest final timestamp
                latest_forecast_df = max(forecasts, key=lambda df: df['timestamp'].max()).copy()

                # ✅ Clean final forecast before saving
                latest_forecast_df = latest_forecast_df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

                forecast_file = os.path.join(output_dir, f"{sensor}_forecast.csv")
                latest_forecast_df.to_csv(forecast_file, index=False)
                print(f"💾 Saved forecast to {forecast_file}")
                print(f"📉 RMSE for {sensor} (latest window): {rmse_list[-1]:.3f}")

                results_summary.append({
                    'sensor': sensor,
                    'avg_rmse': rmse_list[-1],
                    'backtest_runs': 1
                })

        except Exception as e:
            print(f"⚠️ Failed to model {sensor}: {e}")

    # ✅ Save forecast summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "forecast_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📊 Forecast summary saved to {summary_path}")
