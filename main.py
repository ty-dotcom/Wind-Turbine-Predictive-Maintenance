import numpy as np
import streamlit as st
import pandas as pd

from ModelManagement import load_data, load_model

COLUMNS = ["gearbox_oil_temp",
           "gearbox_bearing_temp",
           "vibration_x",
           "vibration_y",
           "vibration_z",
           "oil_pressure",
           "particle_count"]

SLIDERS_CONFIG = {
    "gearbox_oil_temp": {"min": 40.0, "max": 100.0, "step": 0.1, "format": "%.1f"},
    "gearbox_bearing_temp": {"min": 50.0, "max": 100.0, "step": 0.1, "format": "%.1f"},
    "vibration_x": {"min": 0.0, "max": 0.05, "step": 0.0001, "format": "%.4f"},
    "vibration_y": {"min": 0.0, "max": 0.04, "step": 0.0001, "format": "%.4f"},
    "vibration_z": {"min": 0.0, "max": 0.06, "step": 0.0001, "format": "%.4f"},
    "oil_pressure": {"min": 0.0, "max": 5.0, "step": 0.01, "format": "%.2f"},
    "particle_count": {"min": 100.0,"max": 2000.0, "step": 1.0, "format": "%.0f"},
}

LABEL_MAP = {
    0: "Healthy!",
    1: "Warning!",
}
st.set_page_config(
    page_title="Wind Turbine Gearbox Predictive Maintenance System",
    page_icon=":material/cyclone:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Wind Turbine Gearbox Predictive Maintenance System")

@st.cache_resource
def get_model():
    data = load_data()
    return load_model(data)

with st.spinner("Loading data and models..."):
    model = get_model()

st.session_state['model_name'] = model.get_name()
st.session_state['prediction'] = []

st.subheader(f"Active Model: {model.get_name()}")

container = st.container()

def prediction_color_map(val):
    color = "green" if val == "Healthy!" else "yellow"
    return f"background-color: {color}"

def make_prediction(value):
    """Predicts Normal/Anomaly and shows the status for the prediction."""
    predictions = model.predict(value)
    predictions_mapping = [LABEL_MAP[pred] for pred in predictions]
    if len(predictions_mapping) == 1:
        if predictions_mapping[0] == "Warning!":
            container.warning("Warning!", icon=":material/e911_emergency:")
        else:
            container.success("Healthy!", icon=":material/check:")
    return predictions_mapping

def display_upload_classifier():
    """Upload a CSV/Excel file with sensor data and predicts each row."""
    file = st.file_uploader("Upload a CSV/Excel file", type=["csv", "tsv", "xlsx", "xls", "xlsm", "xlsb"])

    if file is None:
        st.session_state.pop("predicted_df", None)
        return

    try:
        if file.type in ("csv", "tsv", "text/csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        container.error(f"Failed to read {file}: {e}")
        return

    if "timestamp" in df.columns:
        df = df.drop("timestamp", axis=1)

    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        container.error(f"Missing columns: {missing}")
        return

    dataframe = container.dataframe(df.head(100))
    if st.button("Predict"):
        df["prediction"] = make_prediction(df[COLUMNS])
        st.session_state["predicted_df"] = df.copy()

    if "predicted_df" not in st.session_state:
        return
    df = st.session_state["predicted_df"]
    anomaly_count = int(np.sum(df["prediction"] == "Warning!"))
    normal_count = int(np.sum(df["prediction"] == "Healthy!"))
    total = len(df)
    container.subheader("Prediction Summary")
    col1, col2, col3 = container.columns(3)

    col1.metric("Total rows", total)
    col2.metric("Anomalies found", anomaly_count)
    col3.metric("Healthy rows", normal_count)

    if total:
        container.caption(
            f"Anomaly rate: **{anomaly_count / total:.4f}%**"
        )

    container.divider()
    show_anomalies = st.toggle(
        "Show Anomalies",
        value=True,
        help="Toggle off to see all rows including anomalies."
    )

    if show_anomalies:
        anomalies_df = df[df["prediction"] == "Warning!"]
        if len(anomalies_df) == 0:
            container.success(
                "No anomalies detected. Toggle off to see all rows.",
                    icon=":material/check:")
        else:
            dataframe.dataframe(
                anomalies_df.style.map(prediction_color_map, subset=["prediction"])
            )
    else:
        pd.set_option("styler.render.max_elements", 2102400)
        dataframe.dataframe(
            df.style.map(prediction_color_map, subset=["prediction"])
        )

def display_classifier():
    """Displays sliders for each sensor reading, handling the prediction once the button is pressed."""
    vals = {}
    for col in COLUMNS:
        config = SLIDERS_CONFIG[col]
        vals[col] = st.slider(
            label=col.replace("_", " ").title(),
            min_value=config["min"],
            max_value=config["max"],
            step=config["step"],
            format=config["format"],
        )


    if st.button("Predict"):
        x = pd.DataFrame([vals], columns=COLUMNS)
        make_prediction(x)

def change_model():
    model.change_model(st.session_state["model_name"])

with st.sidebar:
    select_model = st.selectbox(
        "Choose a model",
        ["XGBClassifier", "IsolationForest", "LGBMClassifier"],
                key="model_name", on_change=change_model
    )

    method = st.selectbox("Choose Input Method",
                          options=["Upload", "Manual"],
                          key="method"
                          )
    if method == "Upload":
        display_upload_classifier()
    else:
        display_classifier()