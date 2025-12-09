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

# -----------------------
# CONFIG
# -----------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/sic/make/sensor"
TOPIC_OUTPUT = "iot/sic/make/output"
MODEL_PATH = "iot_temp_model.pkl"

MAX_POINTS = 200
ANOMALY_Z_THRESHOLD = 3.0

# -----------------------
# Session State
# -----------------------
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

# -----------------------
# Load ML Model
# -----------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model '{path}': {e}")
        return None

model = load_model(MODEL_PATH)
st.session_state.model_loaded = model is not None

# -----------------------
# MQTT Worker (FAST RESPONSE)
# -----------------------
def mqtt_worker(broker, port, topic_sensor, topic_output, in_q, out_q):
    client = mqtt.Client()

    def _on_connect(c, userdata, flags, rc):
        print("MQTT connected with rc:", rc)
        if rc == 0:
            c.subscribe(topic_sensor)

    def _on_message(c, userdata, msg):
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
            in_q.put({
                "ts": (datetime.utcnow() + timedelta(hours=7)).isoformat(),
                "topic": msg.topic,
                "payload": data
            })
        except Exception as e:
            print("Failed parse incoming msg:", e)

    client.on_connect = _on_connect
    client.on_message = _on_message

    while True:
        try:
            client.connect(broker, port, keepalive=60)
            client.loop_start()

            while True:
                try:
                    item = out_q.get(timeout=0.05)   # <— FAST RESPONSE
                except queue.Empty:
                    item = None

                if item is not None:
                    client.publish(
                        item["topic"],
                        item["payload"],
                        qos=0,
                        retain=False
                    )

        except Exception as e:
            print("MQTT worker error:", e)
            try:
                client.loop_stop()
                client.disconnect()
            except:
                pass
            time.sleep(1)

# Start MQTT Thread
if not st.session_state.mqtt_worker_started:
    t = threading.Thread(
        target=mqtt_worker,
        args=(MQTT_BROKER, MQTT_PORT, TOPIC_SENSOR, TOPIC_OUTPUT,
              st.session_state.mqtt_in_q, st.session_state.mqtt_out_q),
        daemon=True
    )
    t.start()
    st.session_state.mqtt_worker_started = True
    time.sleep(0.1)

# -----------------------
# Process Incoming Data
# -----------------------
def process_incoming():
    q = st.session_state.mqtt_in_q

    while not q.empty():
        item = q.get()
        payload = item["payload"]
        ts = item["ts"]

        try: temp = float(payload.get("temp"))
        except: temp = None
        try: hum = float(payload.get("hum"))
        except: hum = None

        row = {"ts": ts, "temp": temp, "hum": hum}

        pred = None
        conf = None
        anomaly = False

        if model is not None and temp is not None and hum is not None:
            X = [[temp, hum]]

            try:
                pred = model.predict(X)[0]
            except:
                pred = None

            try:
                conf = float(np.max(model.predict_proba(X)))
            except:
                conf = None

        row.update({"pred": pred, "conf": conf, "anomaly": anomaly})
        st.session_state.last = row
        st.session_state.logs.append(row)

        if len(st.session_state.logs) > 5000:
            st.session_state.logs = st.session_state.logs[-5000:]

        # SEND ML prediction → ESP32
        if pred is not None:
            st.session_state.mqtt_out_q.put({
                "topic": TOPIC_OUTPUT,
                "payload": str(pred)
            })

process_incoming()

# -----------------------
# UI SETUP
# -----------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")
st.title("🔥 IoT ML Realtime Dashboard — FINAL FAST VERSION")

# Auto-refresh (fast)
st_autorefresh(interval=500, limit=None, key="refresh")

left, right = st.columns([1,2])

# -----------------------
# LEFT PANEL
# -----------------------
with left:
    st.header("Connection Status")
    st.metric("MQTT Connected", "Yes" if len(st.session_state.logs) > 0 else "No")

    st.markdown("### Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: {last['ts']}")
        st.write(f"Temp : {last['temp']} °C")
        st.write(f"Hum  : {last['hum']} %")
        st.write(f"Pred : {last['pred']}")
        st.write(f"Conf : {last['conf']}")
        st.write(f"Anom : {last['anomaly']}")
    else:
        st.info("Waiting for data...")

    st.markdown("### Manual Control (FAST)")
    col1, col2 = st.columns(2)
    if col1.button("ALERT_ON"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_ON"})
    if col2.button("ALERT_OFF"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_OFF"})

    st.markdown("### Anomaly Settings")
    st.session_state.anomaly_window = st.slider(
        "Window Size", 5, 200, st.session_state.anomaly_window
    )

    # -----------------------
    # DOWNLOAD CSV (RESTORED)
    # -----------------------
    st.markdown("### Download Logs")
    if st.button("Download CSV"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV file",
                csv,
                file_name=f"iot_logs_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
            )
        else:
            st.warning("No logs yet")

# -----------------------
# RIGHT PANEL
# -----------------------
with right:
    st.header(f"Live Chart (Last {MAX_POINTS} Points)")
    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])

    if not df_plot.empty:
        fig = go.Figure()

        # Temperature
        fig.add_trace(go.Scatter(
            x=df_plot["ts"],
            y=df_plot["temp"],
            mode="lines+markers",
            name="Temperature (°C)"
        ))

        # Humidity (FIXED)
        fig.add_trace(go.Scatter(
            x=df_plot["ts"],
            y=df_plot["hum"],
            mode="lines+markers",
            name="Humidity (%)",
            yaxis="y2"
        ))

        fig.update_layout(
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right")
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(50))
