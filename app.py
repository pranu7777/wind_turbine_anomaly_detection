import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from statsmodels.tsa.statespace.sarimax import SARIMAX
from feature_utils import extract_features_and_labels
from forecast_utils import forecast_all_sensors

app = Flask(__name__)

# ------------------------------
# Setup
# ------------------------------
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
FORECAST_FOLDER = "sensor_forecasts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FORECAST_FOLDER, exist_ok=True)

# Load models
MODEL = joblib.load(os.path.join(MODEL_FOLDER, "xgboost_model.pkl"))
TTF_MODEL = joblib.load(os.path.join(MODEL_FOLDER, "lightgbm_ttf_model.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_FOLDER, "sensor_columns.pkl"))
TTF_FEATURES = joblib.load(os.path.join(MODEL_FOLDER, "ttf_feature_list.pkl"))

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' not in request.files:
        return " No file uploaded", 400

    file = request.files['csv_file']
    if file.filename == '':
        return " No file selected", 400

    try:
        df = pd.read_csv(file)

        for col in ['true_label', 'predicted_label', 'probability', 'risk']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURES].astype(float)
        y_pred = MODEL.predict(df)
        y_prob = MODEL.predict_proba(df)[:, 1]

        df['true_label'] = 'Uploaded'
        df['predicted_label'] = y_pred
        df['probability'] = y_prob

        def classify_risk(prob):
            if prob >= 0.85:
                return "🔴 High Risk"
            elif prob >= 0.5:
                return "🟡 Moderate Risk"
            else:
                return "🟢 Safe"

        df['risk'] = df['probability'].apply(classify_risk)

        safe = int((df['risk'] == "🟢 Safe").sum())
        moderate = int((df['risk'] == "🟡 Moderate Risk").sum())
        high = int((df['risk'] == "🔴 High Risk").sum())

        risk_counts = df['risk'].value_counts()
        dominant = str(risk_counts.idxmax()) if not risk_counts.empty else "🟢 Safe"
        dominant_icon = dominant[0]
        dominant_label = dominant[2:]

        df.to_csv(os.path.join(UPLOAD_FOLDER, file.filename), index=False)
        df.to_csv(os.path.join(UPLOAD_FOLDER, "last_uploaded.csv"), index=False)

        return render_template("result.html",
                               title="Anomaly Classification",
                               filename=file.filename,
                               total=len(df),
                               anomalies=int(sum(y_pred)),
                               normal=int(len(df) - sum(y_pred)),
                               max_prob=f"{y_prob.max():.2%}",
                               avg_prob=f"{y_prob.mean():.2%}",
                               columns=df.columns.tolist(),
                               data=df.head(20).values.tolist(),
                               probabilities=[round(p, 3) for p in y_prob.tolist()],
                               chart=True,
                               risk_level_summary={
                                   "🟢 Safe": safe,
                                   "🟡 Moderate Risk": moderate,
                                   "🔴 High Risk": high
                               },
                               dominant_risk=dominant_label,
                               dominant_risk_icon=dominant_icon)

    except Exception as e:
        import traceback
        print(" Upload Error:\n", traceback.format_exc())
        return render_template("result.html", title="Error", columns=["Error"], data=[[str(e)]], chart=False)
@app.route('/upload_ttf', methods=['POST'])
def upload_file_for_ttf():
    if 'csv_file' not in request.files:
        return " No file uploaded", 400

    file = request.files['csv_file']
    if file.filename == '':
        return " No file selected", 400

    try:
        df = pd.read_csv(file)

        # Save uploaded file (optional but useful)
        filename = file.filename
        df.to_csv(os.path.join(UPLOAD_FOLDER, filename), index=False)
        df.to_csv(os.path.join(UPLOAD_FOLDER, "last_uploaded.csv"), index=False)

        # Ensure timestamp exists
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='10s')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")

        # Feature extraction for TTF
        X_ttf, _, _ = extract_features_and_labels(df, label_required=False)
        for col in TTF_FEATURES:
            if col not in X_ttf.columns:
                X_ttf[col] = 0
        X_ttf = X_ttf[TTF_FEATURES]

        # Predict
        predictions = TTF_MODEL.predict(X_ttf)
        df['predicted_minutes_to_failure'] = np.abs(predictions).round(2)

        df_result = df[['timestamp', 'predicted_minutes_to_failure']]

        return render_template("result.html",
                               title="TTF Prediction from Upload",
                               filename=filename,
                               columns=df_result.columns.tolist(),
                               data=df_result.head(20).values.tolist(),
                               chart=False)

    except Exception as e:
        import traceback
        print(" TTF Upload Prediction Error:\n", traceback.format_exc())
        return render_template("result.html",
                               title="Error",
                               columns=["Error"],
                               data=[[str(e)]],
                               chart=False)

@app.route('/forecast')
def view_forecast():
    try:
        summary_path = os.path.join(FORECAST_FOLDER, "forecast_summary.csv")
        if not os.path.exists(summary_path):
            return " Forecast summary file not found.", 404

        df = pd.read_csv(summary_path)

        if df.empty or df.shape[1] == 0:
            return render_template("result.html",
                                   title="Sensor Forecast Summary",
                                   columns=["Info"],
                                   data=[["⚠️ The forecast_summary.csv file is empty or invalid."]],
                                   chart=False,
                                   show_sensor_form=True)

        return render_template("result.html",
                               title="Sensor Forecast Summary",
                               filename="forecast_summary.csv",
                               columns=df.columns.tolist(),
                               data=df.head(20).values.tolist(),
                               chart=False,
                               show_sensor_form=True)

    except Exception as e:
        import traceback
        print(" Forecast Display Error:\n", traceback.format_exc())
        return render_template("result.html",
                               title="Error",
                               columns=["Error"],
                               data=[[str(e) or "Unknown forecast error"]],
                               chart=False)


@app.route('/forecast/sensor', methods=['POST'])
def forecast_sensor_redirect():
    sensor_number = request.form.get('sensor_number')
    return redirect(url_for('view_sensor_forecast', sensor_number=sensor_number))


@app.route('/forecast/sensor/<sensor_number>')
def view_sensor_forecast(sensor_number):
    try:
        filename = f"sensor_{sensor_number}_avg_avg_forecast.csv"
        path = os.path.join(FORECAST_FOLDER, filename)

        if not os.path.exists(path):
            return render_template("result.html",
                                   title="Sensor Forecast Error",
                                   columns=["Error"],
                                   data=[[f" File not found for sensor {sensor_number}"]],
                                   chart=False)

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Prepare chart data
        chart_data = {
            "labels": df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "actual": df['actual'].round(2).tolist(),
            "forecast": df['forecast'].round(2).tolist()
        }

        return render_template("result.html",
                               title=f"Forecast for Sensor {sensor_number}",
                               filename=filename,
                               chart_data=chart_data,
                               chart_type="line")

    except Exception as e:
        import traceback
        print(" Sensor Forecast Display Error:\n", traceback.format_exc())
        return render_template("result.html",
                               title="Error",
                               columns=["Error"],
                               data=[[str(e) or "Unknown forecast error"]],
                               chart=False)



if __name__ == '__main__':
    app.run(debug=True)
