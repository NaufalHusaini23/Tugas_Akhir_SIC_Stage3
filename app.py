# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# =====================================================
#  DARK MODE – GLASSMORPHISM – BLUE / PURPLE AURORA UI
# =====================================================
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top left, #1a1c2c, #0b0c17 70%);
}

.block-container {
    padding-top: 2rem;
}

/* HEADERS */
h1, h2, h3, h4 {
    color: #e0e6ff !important;
    text-shadow: 0px 0px 8px rgba(70, 70, 255, 0.3);
}

/* GLASS CARD METRIC */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 4px 25px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
}

/* PANELS (LEFT + RIGHT) */
.css-1kyxreq, .css-12w0qpk {
    background: rgba(255,255,255,0.04);
    border-radius: 15px;
    padding: 20px !important;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    backdrop-filter: blur(12px);
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #5b2be0, #3d7bff);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
    transition: 0.2s;
    box-shadow: 0 0 12px rgba(110, 90, 255, 0.6);
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 18px rgba(120,110,255,0.9);
}

/* DOWNLOAD BUTTON */
.stDownloadButton>button {
    background: linear-gradient(90deg, #3b5cff, #6d29ff);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    font-weight: 600;
    transition: 0.2s;
    box-shadow: 0 0 15px rgba(90, 70, 250, 0.7);
}

.stDownloadButton>button:hover {
    transform: scale(1.05);
}

/* CHART CARD */
.stPlotlyChart {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}

/* TABLE */
.dataframe {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px;
    color: #d8d8f0 !important;
}

/* TEXT */
p, label, span, div {
    color: #cfd2ff !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#5b2be0, #3d7bff);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# CONFIG (TIDAK DIUBAH)
# =====================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/sic/make/sensor"
TOPIC_OUTPUT = "iot/sic/make/output"
MODEL_PATH = "iot_temp_model.pkl"

MAX_POINTS = 200
ANOMALY_Z_THRESHOLD = 3.0

# =====================================================
# Session State (TIDAK DIUBAH)
# =====================================================
if "mqtt_in_q" not in st.session_state:
    st.session_state.mqtt_in_q = queue.Queue()
if "mqtt_out_q" not in st.session_state:
    st.session_state.mqtt_out_q = queue.Queue()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_worker_started" not in st.session_state:
    st.session_state.mqtt_worker_started = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "anomaly_window" not in st.session_state:
    st.session_state.anomaly_window = 30


# =====================================================
# MODEL LOADING (TIDAK DIUBAH)
# =====================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except:
        st.error("Model gagal dimuat!")
        return None

model = load_model(MODEL_PATH)
st.session_state.model_loaded = model is not None


# =====================================================
# MQTT WORKER (TIDAK DIUBAH)
# =====================================================
def mqtt_worker(broker, port, topic_sensor, topic_output, in_q, out_q):
    client = mqtt.Client()

    def _on_connect(c, userdata, flags, rc):
        if rc == 0:
            c.subscribe(topic_sensor)

    def _on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            in_q.put({
                "ts": (datetime.utcnow() + timedelta(hours=7)).isoformat(),
                "topic": msg.topic,
                "payload": payload
            })
        except:
            pass

    client.on_connect = _on_connect
    client.on_message = _on_message

    while True:
        try:
            client.connect(broker, port)
            client.loop_start()

            while True:
                try:
                    item = out_q.get(timeout=0.5)
                    client.publish(item["topic"], item["payload"])
                except queue.Empty:
                    pass

        except:
            try: client.loop_stop()
            except: pass
            time.sleep(2)


# Start thread
if not st.session_state.mqtt_worker_started:
    threading.Thread(
        target=mqtt_worker,
        args=(MQTT_BROKER, MQTT_PORT, TOPIC_SENSOR, TOPIC_OUTPUT,
              st.session_state.mqtt_in_q, st.session_state.mqtt_out_q),
        daemon=True
    ).start()
    st.session_state.mqtt_worker_started = True
    time.sleep(0.1)


# =====================================================
# PROCESS INCOMING (TIDAK DIUBAH)
# =====================================================
def process_incoming():
    q = st.session_state.mqtt_in_q

    while not q.empty():
        item = q.get()
        payload = item["payload"]

        try: temp = float(payload.get("temp"))
        except: temp = None
        try: hum = float(payload.get("hum"))
        except: hum = None

        row = {"ts": item["ts"], "temp": temp, "hum": hum}

        pred, conf, anomaly = None, None, False

        if model and temp and hum:
            X = [[temp, hum]]
            try: pred = model.predict(X)[0]
            except: pass

            try: conf = float(np.max(model.predict_proba(X)))
            except: pass

            temps = [r["temp"] for r in st.session_state.logs if r.get("temp")]
            window = temps[-st.session_state.anomaly_window:]
            if len(window) >= 5:
                mean = float(np.mean(window))
                std = float(np.std(window))
                if std > 0:
                    z = abs((temp - mean) / std)
                    if z >= ANOMALY_Z_THRESHOLD:
                        anomaly = True

        row.update({"pred": pred, "conf": conf, "anomaly": anomaly})
        st.session_state.last = row
        st.session_state.logs.append(row)

        if len(st.session_state.logs) > 5000:
            st.session_state.logs = st.session_state.logs[-5000:]

        # Kirim ML ke ESP
        if pred is not None:
            st.session_state.mqtt_out_q.put({
                "topic": TOPIC_OUTPUT,
                "payload": str(pred)
            })

process_incoming()


# =====================================================
# UI – TIDAK ADA FUNGSI DIUBAH, HANYA TAMPILAN
# =====================================================
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")
st.title("🌌 IoT ML Realtime Dashboard — Smart Environment Monitoring")

if st.session_state.model_loaded:
    st.success("ML Model Loaded")
else:
    st.warning("Model not loaded!")

st_autorefresh(interval=2000)

left, right = st.columns([1, 2])

# LEFT PANEL
with left:
    st.header("📡 Connection Status")
    st.metric("MQTT Connected", "Yes" if st.session_state.logs else "No")

    st.markdown("### 📊 Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"**Time:** {last['ts']}")
        st.write(f"**Temperature:** {last['temp']} °C")
        st.write(f"**Humidity:** {last['hum']} %")
        st.write(f"**Prediction:** {last['pred']}")
        st.write(f"**Confidence:** {last['conf']}")
        st.write(f"**Anomaly:** {last['anomaly']}")
    else:
        st.info("Waiting for data...")

    st.markdown("### 🔧 Manual Control")
    col1, col2 = st.columns(2)
    if col1.button("Send ALERT_ON"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_ON"})
    if col2.button("Send ALERT_OFF"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_OFF"})

    st.markdown("### ⚠ Anomaly Settings")
    w = st.slider("Window Size", 5, 200, st.session_state.anomaly_window)
    st.session_state.anomaly_window = w

    # DOWNLOAD CSV
    st.markdown("### 📥 Download Logs")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="iot_logs.csv",
            mime="text/csv"
        )
    else:
        st.write("No logs yet")

# RIGHT PANEL
with right:
    st.header("📈 Live Chart")
    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])

    if not df_plot.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["temp"],
            mode="lines+markers", name="Temperature (°C)"
        ))
        fig.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["hum"],
            mode="lines+markers", name="Humidity (%)",
            yaxis="y2"
        ))

        fig.update_layout(
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right")
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📝 Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(50))

process_incoming()
