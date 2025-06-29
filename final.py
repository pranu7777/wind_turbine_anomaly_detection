import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold, cross_val_score


# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_PATH = r'D:\wind_turbine_anomaly\Data'
FARMS = ['Wind Farm A', 'Wind Farm B', 'Wind Farm C']
MODEL_SAVE_PATH = 'models'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


# -------------------------------
# TIME-TO-FAILURE LABEL GENERATION
# -------------------------------
def add_time_to_failure_labels(df):
    """
    Add 'time_to_failure' column to each row in the DataFrame.
    For each row, calculate time (in minutes) until the next anomaly event.
    Rows without a future anomaly are dropped.

    Parameters:
    df (pd.DataFrame): DataFrame with 'timestamp' and 'event_label' columns.

    Returns:
    pd.DataFrame: Original DataFrame with an added 'time_to_failure' column.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("Timestamp column required for time-to-failure.")

    # Ensure rows are in chronological order
    df = df.sort_values('timestamp')

    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get the indices where anomalies occur
    failure_indices = df.index[df['event_label'] == 'anomaly'].tolist()

    # Initialize time_to_failure column with NaN
    df['time_to_failure'] = np.nan

    # For each row, find the time difference to the next anomaly
    for idx in df.index:
        next_failure = next((f for f in failure_indices if f > idx), None)
        if next_failure:
            df.at[idx, 'time_to_failure'] = (
                df.loc[next_failure, 'timestamp'] - df.loc[idx, 'timestamp']
            ).total_seconds() / 60  # time in minutes

    # Remove rows that don't have a future failure
    return df.dropna(subset=['time_to_failure'])


def load_training_data():
    """
    Load and combine training data from multiple farms and sensor files.
    Aligns all datasets with timestamps, filters non-train rows, and validates column consistency.

    Returns:
    pd.DataFrame: Concatenated and labeled training data with consistent structure.
    """
    all_data = []
    global_start_time = pd.to_datetime('2020-01-01 00:00:00')  # Initial timestamp
    ref_columns = None  # Used to enforce consistent columns across files

    for farm in FARMS:
        farm_dir = os.path.join(BASE_PATH, farm)
        dataset_dir = os.path.join(farm_dir, 'datasets')
        event_info_path = os.path.join(farm_dir, 'comma_event_info.csv')

        # Skip farm if event info is missing
        if not os.path.exists(event_info_path):
            print(f" Missing event info for {farm}")
            continue

        event_info = pd.read_csv(event_info_path)

        for file in os.listdir(dataset_dir):
            if not file.endswith('.csv'):
                continue  # Skip non-CSV files

            file_path = os.path.join(dataset_dir, file)

            # Try reading file preview to catch errors early
            try:
                preview = pd.read_csv(file_path, nrows=5)
                if preview.shape[1] > 1000:
                    print(f" Skipped {file}: too many columns {preview.shape}")
                    continue

                # Load full data if preview looks good
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

                # Skip files that are too small or have too many columns
                if df.shape[1] > 1000 or df.shape[0] < 5:
                    print(f" Skipped {file}: suspicious shape {df.shape}")
                    continue

            except Exception as e:
                print(f" Error reading {file}: {e}")
                continue

            # Extract event label using the event ID
            event_id = file.replace('.csv', '').replace('comma_', '')
            label_row = event_info[event_info['event_id'].astype(str) == event_id]
            if label_row.empty:
                continue  # Skip files with no matching label

            # Filter only training rows
            df = df[df['train_test'] == 'train']
            if df.empty:
                continue

            # Ensure consistent columns across all training data
            df_columns_set = set(df.columns) - {'timestamp', 'event_label', 'farm', 'source_file'}
            if ref_columns is None:
                ref_columns = df_columns_set
            elif df_columns_set != ref_columns:
                print(f" Skipped {file}: column mismatch with reference")
                continue

            # Generate synthetic timestamps for alignment
            df['timestamp'] = pd.date_range(start=global_start_time, periods=len(df), freq='10s')
            global_start_time = df['timestamp'].iloc[-1] + pd.Timedelta(seconds=10)

            # Add label and metadata
            df['event_label'] = label_row['event_label'].values[0]
            df['farm'] = farm
            df['source_file'] = file

            all_data.append(df)

    # Final validation
    if not all_data:
        raise ValueError(" No valid training data found.")

    print(f" Concatenating {len(all_data)} dataframes with matching columns...")

    # Combine and sort all valid data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    return combined




# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

def extract_features_and_labels(df):
    base_cols = [col for col in df.columns if col.startswith('sensor_')]
    df = df.copy()
    feature_frames = []

    for col in base_cols:
        feature_frames.append(df[col].diff().rename(f"{col}_diff"))
        feature_frames.append(df[col].rolling(3).mean().rename(f"{col}_roll_mean"))
        feature_frames.append(df[col].rolling(3).std().fillna(0).rename(f"{col}_roll_std"))
        feature_frames.append(df[col].rolling(5).min().rename(f"{col}_roll_min"))
        feature_frames.append(df[col].rolling(5).max().rename(f"{col}_roll_max"))
        feature_frames.append((df[col] - df[col].shift(3)).rename(f"{col}_momentum"))
        feature_frames.append(df[col].ewm(span=3).mean().rename(f"{col}_ewm_mean"))

    # Ratio features
    if 'sensor_1' in df.columns and 'sensor_2' in df.columns:
        df['sensor_ratio_1_2'] = df['sensor_1'] / (df['sensor_2'] + 1e-5)

    df_engineered = pd.concat([df] + feature_frames, axis=1)

    # Feature filtering
    feature_cols = [col for col in df_engineered.columns if col.startswith('sensor_')]
    X = df_engineered[feature_cols].fillna(0)

    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=drop_cols)

    # Remove near-zero variance
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    y = df_engineered['event_label'].map({'normal': 0, 'anomaly': 1})
    return X, y, X.columns.tolist()

# -------------------------------
# MODEL TRAINING
# -------------------------------
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import plot_importance
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve
from xgboost import plot_importance

def train_and_evaluate_models(X_train, y_train, X_test, y_test, tune_xgb=True):
    results = []
    predictions = {}

    # Handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    print("\n Training RandomForest...")
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    results.append({
        'Model': 'RandomForest',
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'F1 Score': f1_score(y_test, rf_pred),
        'ROC AUC': roc_auc_score(y_test, rf_prob)
    })
    predictions['RandomForest'] = {'y_pred': rf_pred, 'y_prob': rf_prob}
    joblib.dump(rf_model, "models/randomforest_model.pkl")

    # XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    if tune_xgb:
        print("\n🔍 Tuning XGBoost...")
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.02, 0.05],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'scale_pos_weight': [scale_pos_weight]
        }

        search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring='f1',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        xgb_model = search.best_estimator_
        print(f" Best Params: {search.best_params_}")
    else:
        xgb_model.set_params(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight
        )
        xgb_model.fit(X_train, y_train)

    # Predict and threshold
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Optional: threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, xgb_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f" Best threshold for F1: {best_thresh:.3f}")

    xgb_pred = (xgb_prob > best_thresh).astype(int)

    results.append({
        'Model': 'XGBoost (Tuned)',
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1 Score': f1_score(y_test, xgb_pred),
        'ROC AUC': roc_auc_score(y_test, xgb_prob)
    })
    predictions['XGBoost'] = {'y_pred': xgb_pred, 'y_prob': xgb_prob}
    joblib.dump(xgb_model, "models/xgboost_model.pkl")

    print("\n XGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, xgb_pred))

    print(" Plotting XGBoost Feature Importance...")
    plot_importance(xgb_model, max_num_features=20)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results), predictions


# -------------------------------
# TIME-TO-FAILURE REGRESSION MODEL
# -------------------------------
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE)
    Excludes any zero values in the true targets to avoid division errors.

    Parameters:
    y_true (array-like): Ground truth (actual) values.
    y_pred (array-like): Predicted values.

    Returns:
    float: MAPE score as a percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Mask out entries where y_true is 0 to avoid division by zero
    mask = y_true != 0
    
    # Compute average of absolute percentage errors
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_time_to_failure(X, y):
    """
    Train and evaluate regression models to predict Time-To-Failure (TTF).

    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector representing time-to-failure in minutes.

    Saves:
    - Trained models (Gradient Boosting and LightGBM) to disk as .pkl files.
    - Prints evaluation metrics (MAE, RMSE, R², MAPE) for each model.
    """
    
    # Dictionary of regression models to train
    models = {
        'GradientBoosting': GradientBoostingRegressor(),
        'LightGBM': lgb.LGBMRegressor()
    }

    for name, model in models.items():
        print(f"\n Training TTF model: {name}")

        # Fit model to training data
        model.fit(X, y)

        # Predict using the same training data (for initial evaluation)
        y_pred = model.predict(X)

        # Evaluate model using common regression metrics
        mae = mean_absolute_error(y, y_pred)                      # Mean Absolute Error
        rmse = mean_squared_error(y, y_pred, squared=False)       # Root Mean Squared Error
        r2 = r2_score(y, y_pred)                                  # R-squared score
        mape = mean_absolute_percentage_error(y, y_pred)          # Mean Absolute Percentage Error

        # Print evaluation results
        print(f" {name} MAE: {mae:.2f}")
        print(f" {name} RMSE: {rmse:.2f}")
        print(f" {name} R² Score: {r2:.3f}")
        print(f" {name} MAPE: {mape:.2f}%")

        # Save trained model to disk for future use
        joblib.dump(model, f"{MODEL_SAVE_PATH}/{name.lower()}_ttf_model.pkl")



# -------------------------------
# VISUALIZATION
# -------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_df):
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis')

    # Add accuracy values on top of bars
    for i, row in results_df.iterrows():
        ax.text(i, row['Accuracy'] + 0.002, f"{row['Accuracy']:.3f}",
                color='black', ha='center', fontweight='bold')

    plt.title("Model Comparison - Validation Accuracy")
    plt.ylim(results_df['Accuracy'].min() - 0.01, results_df['Accuracy'].max() + 0.02)
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------
# PLACEHOLDER FOR SENSOR FORECASTING
# -------------------------------

def forecast_sensor(sensor_series):
    print(" Forecasting not implemented yet.")
    pass


# -------------------------------
# SAVE FINAL PREDICTIONS
# -------------------------------
import pandas as pd

def save_predictions(X_test, y_test, preds, filename="xgboost_predictions.csv"):
    # Convert to DataFrame if needed
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

    df = X_test.copy()
    df['true_label'] = y_test.values
    df['predicted_label'] = preds['y_pred']
    df['probability'] = preds['y_prob']

    df.to_csv(filename, index=False)
    print(f" Predictions saved to {filename}")
    

def forecast_sensor_lstm(csv_path='combined_fuel_stock_data.csv', n_steps=30, n_future=10, sensor_prefix='sensor_'):
    print(f"\n Loading dataset: {csv_path}")
    if not os.path.exists(csv_path):
        print(f" File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Filter only sensor columns
    sensor_cols = [col for col in df.columns if col.startswith(sensor_prefix)]
    if not sensor_cols:
        print(" No sensor columns found in the dataset.")
        return

    df_sensors = df[sensor_cols].copy()

    print(f" Found {len(sensor_cols)} sensor columns")

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_sensors)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled) - n_steps - n_future):
        X.append(scaled[i:i + n_steps])
        y.append(scaled[i + n_steps:i + n_steps + n_future])

    X, y = np.array(X), np.array(y)
    print(f" Data shape: X={X.shape}, y={y.shape}")

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Model
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(n_steps, X.shape[2])),
        Dense(n_future * X.shape[2])
    ])
    model.compile(optimizer='adam', loss='mse')
    print(" Training LSTM model...")
    model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=10, batch_size=64, verbose=1)

    # Forecast on latest input
    last_input = scaled[-n_steps:].reshape(1, n_steps, X.shape[1])
    prediction = model.predict(last_input)[0].reshape(n_future, -1)
    prediction_inverse = scaler.inverse_transform(prediction)

    # Plot one sensor as example
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_future), prediction_inverse[:, 0], label=f'Forecasted {sensor_cols[0]}')
    plt.title(f'LSTM Forecast for {sensor_cols[0]} - Next {n_future} Steps')
    plt.xlabel('Future Step')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(" Forecast complete.")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def forecast_sensor_lstm_from_df(df, sensor_name='sensor_1', forecast_horizon=50, sequence_length=30):
    if sensor_name not in df.columns:
        print(f" Sensor {sensor_name} not found.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    series = df[sensor_name].fillna(method="ffill").fillna(method="bfill").values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_series) - forecast_horizon):
        X.append(scaled_series[i:i+sequence_length])
        y.append(scaled_series[i+sequence_length:i+sequence_length+forecast_horizon])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        print(" Not enough data to train LSTM.")
        return

    # Model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    print(f" Training LSTM model for {sensor_name}...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    # Forecast next values
    last_seq = scaled_series[-sequence_length:].reshape(1, sequence_length, 1)
    prediction_scaled = model.predict(last_seq)[0]
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    # Plot
    future_index = pd.date_range(start=df['timestamp'].iloc[-1], periods=forecast_horizon + 1, freq='10s')[1:]
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df[sensor_name], label="Historical")
    plt.plot(future_index, prediction, label="Forecast", linestyle='--', marker='o')
    plt.title(f"{sensor_name} Forecast (Next {forecast_horizon} Steps)")
    plt.xlabel("Time")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE), ignoring zero true values to prevent division errors.
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    
    Returns:
    float: MAPE score in percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def forecast_all_sensors(df, horizon=50, backtest_windows=3, output_dir='sensor_forecasts'):
    """
    Forecast future sensor readings using SARIMAX with rolling-window backtesting.
    
    Parameters:
    df (pd.DataFrame): Time-series dataset containing sensor columns and timestamps.
    horizon (int): Number of time steps to forecast in each backtest window.
    backtest_windows (int): Number of rolling backtest evaluations per sensor.
    output_dir (str): Directory to save forecast outputs and summary.
    
    Outputs:
    - CSV forecast file per sensor
    - CSV summary of forecasting metrics for all sensors
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Convert and sort timestamp column
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').copy()

    # Identify all columns starting with 'sensor_'
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    results_summary = []  # To store performance metrics for each sensor

    for sensor in sensor_cols:
        print(f"\n Forecasting {sensor} using SARIMAX...")

        try:
            # Prepare individual sensor data
            series_df = df[['timestamp', sensor]].dropna().copy()
            series_df['timestamp'] = pd.to_datetime(series_df['timestamp'])
            series_df = series_df.sort_values('timestamp')
            series_df.set_index('timestamp', inplace=True)
            series_df.index.freq = pd.infer_freq(series_df.index)  # Infer frequency if missing

            series = series_df[sensor]
            rmse_list, r2_list, mape_list = [], [], []  # Metrics
            forecasts = []  # Store forecasts for export

            # Rolling backtest loop
            for i in range(backtest_windows):
                # Define the cutoff point for training
                split_point = int(len(series) * (0.7 - i * 0.1))
                train = series[:split_point]
                test = series[split_point:split_point + horizon]

                # Skip if test size is too small
                if len(test) < horizon:
                    print(f" Not enough data in window {i} for {sensor}")
                    continue

                # Fit SARIMAX model
                model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)

                # Forecast next `horizon` points
                forecast = model_fit.forecast(steps=horizon)

                actual = test[:horizon].values
                predicted = forecast.values

                # Compute metrics
                rmse = mean_squared_error(actual, predicted, squared=False)
                r2 = r2_score(actual, predicted)
                mape = mean_absolute_percentage_error(actual, predicted)

                # Save metrics
                rmse_list.append(rmse)
                r2_list.append(r2)
                mape_list.append(mape)

                # Define time index for forecast output
                if forecasts:
                    last_time = forecasts[-1]['timestamp'].iloc[-1]
                    start_time = last_time + pd.Timedelta(seconds=10)
                else:
                    start_time = test.index[0]

                forecast_times = pd.date_range(start=start_time, periods=horizon, freq='10s')

                # Create DataFrame of forecast results
                forecast_df = pd.DataFrame({
                    'timestamp': forecast_times,
                    'actual': actual,
                    'forecast': predicted
                })

                forecasts.append(forecast_df)

            # Export and log results
            if forecasts:
                forecast_concat = pd.concat(forecasts, ignore_index=True)
                forecast_file = os.path.join(output_dir, f"{sensor}_avg_forecast.csv")
                forecast_concat.to_csv(forecast_file, index=False)
                print(f" Saved forecast to {forecast_file}")
                print(f" Avg RMSE: {np.mean(rmse_list):.3f}, R²: {np.mean(r2_list):.3f}, MAPE: {np.mean(mape_list):.2f}%")

                # Add to summary
                results_summary.append({
                    'sensor': sensor,
                    'avg_rmse': np.mean(rmse_list),
                    'avg_r2': np.mean(r2_list),
                    'avg_mape': np.mean(mape_list),
                    'backtest_runs': len(rmse_list)
                })

        except Exception as e:
            print(f" Failed to model {sensor}: {e}")

    # Save overall summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "forecast_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Forecast summary saved to {summary_path}")



# -------------------------------
# MAIN
# -------------------------------


MODEL_SAVE_PATH = 'models'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  # Create model save directory if not exists

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    print(" Loading data...")
    df = load_training_data()  # Load labeled sensor data from multiple farms

    print(" Extracting and engineering features for classification...")
    X, y, feature_cols = extract_features_and_labels(df)

    # Step 1: Remove Highly Correlated Features 
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    X.drop(columns=drop_cols, inplace=True)
    feature_cols = [f for f in feature_cols if f not in drop_cols]

    print("🔹 Balancing dataset using SMOTEENN...")
    from imblearn.combine import SMOTEENN
    X_bal, y_bal = SMOTEENN().fit_resample(X, y)  # Handle class imbalance

    # ▌Step 2: Feature Selection
    print("🔹 Selecting top features (SelectKBest)...")
    selector = SelectKBest(score_func=f_classif, k=100)  # Select top 100 features
    X_sel = selector.fit_transform(X_bal, y_bal)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_cols[i] for i in selected_indices]

    # Step 3: Model Tuning with RandomizedSearch 
    print("🔹 Tuning XGBoost with RandomizedSearchCV...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 8, 10, 12],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'scale_pos_weight': [1, 2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=20,
        scoring='f1',
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_sel, y_bal)
    best_xgb = random_search.best_estimator_

    print(f" Best Params: {random_search.best_params_}")
    print(f" Best F1 Score from CV: {random_search.best_score_:.4f}")

    # Step 4: Final Evaluation 
    print(" Splitting data for final evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    print(" Training final XGBoost model...")
    best_xgb.fit(X_train, y_train)
    y_pred = best_xgb.predict(X_test)
    y_prob = best_xgb.predict_proba(X_test)[:, 1]  # Get probability of class 1

    # Evaluate and display metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n Final Tuned XGBoost Performance")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Store model evaluation results
    results_df = pd.DataFrame([{
        'Model': 'XGBoost (Tuned)',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
    }])

    predictions = {
        'XGBoost': {
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    }

    # Step 5: Visualizations 
    print("🔹 Visualizing model comparison...")
    plot_model_comparison(results_df)

    print("🔹 Displaying top XGBoost feature importances...")
    plot_importance(best_xgb, max_num_features=20)
    plt.tight_layout()
    plt.show()

    print("🔹 Saving predictions and model artifacts...")
    save_predictions(X_test, y_test, predictions['XGBoost'])

    # Save model and selected feature list
    model_path = os.path.join(MODEL_SAVE_PATH, "xgboost_model.pkl")
    feature_path = os.path.join(MODEL_SAVE_PATH, "sensor_columns.pkl")
    joblib.dump(best_xgb, model_path)
    joblib.dump(selected_feature_names, feature_path)
    print(f" Model saved to: {model_path}")
    print(f" Feature list saved to: {feature_path}")

    # Step 6: Time-to-Failure Regression 
    print("\n Preparing data for Time-to-Failure regression...")
    try:
        df_ttf = add_time_to_failure_labels(df)
        print(" Extracting features for TTF prediction...")
        X_ttf, _, feature_cols_ttf = extract_features_and_labels(df_ttf)
        y_ttf = df_ttf['time_to_failure']
        X_ttf_final = X_ttf[feature_cols_ttf].fillna(0)

        print(" Training regression models for Time-to-Failure...")
        train_time_to_failure(X_ttf_final, y_ttf)

        # Save TTF feature names
        ttf_feature_path = os.path.join(MODEL_SAVE_PATH, "ttf_feature_list.pkl")
        joblib.dump(feature_cols_ttf, ttf_feature_path)
        print(f" TTF feature list saved to: {ttf_feature_path}")

    except Exception as e:
        print(f"Time-to-Failure model skipped: {e}")

    # Step 7: Sensor Forecasting 
    print("\n Running SARIMAX-based sensor forecasting...")
    forecast_all_sensors(df, horizon=50, backtest_windows=3)

    print(" All done.")

if __name__ == '__main__':
    main()    


