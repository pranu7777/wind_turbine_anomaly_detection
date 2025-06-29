import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def extract_features_and_labels(df, label_required=True):
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

    if 'sensor_1' in df.columns and 'sensor_2' in df.columns:
        df['sensor_ratio_1_2'] = df['sensor_1'] / (df['sensor_2'] + 1e-5)

    df_engineered = pd.concat([df] + feature_frames, axis=1)

    feature_cols = [col for col in df_engineered.columns if col.startswith('sensor_')]
    X = df_engineered[feature_cols].fillna(0)

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=drop_cols)

    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    y = None
    if label_required and 'event_label' in df_engineered.columns:
        y = df_engineered['event_label'].map({'normal': 0, 'anomaly': 1})

    return X, y, X.columns.tolist()


def add_time_to_failure_labels(df):
    if 'timestamp' not in df.columns or 'event_label' not in df.columns:
        raise ValueError("Both 'timestamp' and 'event_label' columns are required.")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    failure_indices = df.index[df['event_label'] == 'anomaly'].tolist()
    df['time_to_failure'] = np.nan

    for idx in df.index:
        current_time = df.loc[idx, 'timestamp']
        next_fail_idx = next((i for i in failure_indices if i > idx), None)
        if next_fail_idx is not None:
            failure_time = df.loc[next_fail_idx, 'timestamp']
            minutes_to_failure = (failure_time - current_time).total_seconds() / 60
            df.at[idx, 'time_to_failure'] = max(minutes_to_failure, 0)

    return df.dropna(subset=['time_to_failure'])
