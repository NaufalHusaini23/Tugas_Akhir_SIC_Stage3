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
# CONFIG (tidak diubah)
# -----------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/sic/make/sensor"
TOPIC_OUTPUT = "iot/sic/make/output"
MODEL_PATH = "iot_temp_model.pkl"

MAX_POINTS = 200
ANOMALY_Z_THRESHOLD = 3.0

# -----------------------
# Session State (tidak diubah)
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
# Load ML Model (tidak diubah)
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
# MQTT Worker (tidak diubah)
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
                    item = out_q.get(timeout=0.5)
                except queue.Empty:
                    item = None

                if item is not None:
                    client.publish(
                        item["topic"],
                        item["payload"],
                        qos=int(item.get("qos", 0)),
                        retain=bool(item.get("retain", False))
                    )

        except Exception as e:
            print("MQTT worker error:", e)
            try:
                client.loop_stop()
                client.disconnect()
            except:
                pass
            time.sleep(2)

# Start MQTT thread once
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
# Process Incoming (tidak diubah except send prediction)
# -----------------------
def process_incoming():
    updated = False
    q = st.session_state.mqtt_in_q

    while not q.empty():
        item = q.get()
        payload = item["payload"]
        ts = item["ts"]

        try:
            temp = float(payload.get("temp"))
        except:
            temp = None
        try:
            hum = float(payload.get("hum"))
        except:
            hum = None

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

            temps = [r["temp"] for r in st.session_state.logs if r.get("temp") is not None]
            window = temps[-st.session_state.anomaly_window:] if len(temps) > 0 else []

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

        updated = True

        # SEND ML RESULT BACK (tidak diubah)
        if pred is not None:
            try:
                st.session_state.mqtt_out_q.put({
                    "topic": TOPIC_OUTPUT,
                    "payload": str(pred)
                })
            except Exception as e:
                print("Failed sending prediction:", e)

    return updated

process_incoming()

# -----------------------
# THEME + STYLES (Dark Glassmorphism)
# -----------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

# CSS (glass + dark)
st.markdown(
    """
    <style>
    :root{
      --bg:#0b1020;
      --card:#0f1724b3;
      --muted:#98a0b3;
      --accent1: linear-gradient(135deg,#5ef5ff 0%, #6c63ff 100%);
      --glow: rgba(92, 120, 255, 0.25);
    }
    html, body, [class*="css"]  {
        background: var(--bg);
        color: #e6eef8;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .header {
        background: linear-gradient(90deg, rgba(45,55,72,0.18), rgba(20,24,36,0.1));
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 6px 30px rgba(12,18,34,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.03);
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.03);
        backdrop-filter: blur(6px);
        box-shadow: 0 6px 20px rgba(6,10,20,0.6);
    }
    .stat {
        display:flex;
        align-items:center;
        gap:12px;
    }
    .dot {
        width:12px;
        height:12px;
        border-radius:50%;
        box-shadow: 0 4px 18px rgba(92,120,255,0.12);
    }
    .dot.green { background: #34d399; box-shadow: 0 6px 24px rgba(52,211,153,0.12);}
    .dot.red { background: #fb7185; box-shadow: 0 6px 24px rgba(251,113,133,0.12);}
    .dot.yellow { background: #fbbf24; box-shadow: 0 6px 24px rgba(251,191,36,0.12);}
    .title {
        font-size:20px; font-weight:700; margin:0;
    }
    .sub {
        color: var(--muted); font-size:13px; margin:0;
    }

    /* Buttons */
    div.stButton > button {
      background: linear-gradient(90deg,#6c63ff,#5ef5ff);
      color: #001021;
      font-weight:700;
      border-radius:8px;
      padding:8px 12px;
      border: none;
      box-shadow: 0 8px 20px rgba(92,120,255,0.14);
    }
    div.stButton > button:hover {
      transform: translateY(-2px);
    }

    /* Download button small */
    .download-btn { background: transparent; color: var(--muted); border: 1px solid rgba(255,255,255,0.04); }

    /* Table style fix */
    .stDataFrame table { background: transparent; color: #e6eef8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    f"""
    <div class="header">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="margin:0; color: #cfe9ff;">📡 IoT ML Realtime Dashboard</h2>
          <div style="color:var(--muted); font-size:13px;">Pemantauan Ruang Server — Glass/Dark theme</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:14px; color:var(--muted);">Model:</div>
          <div style="font-weight:700; font-size:16px; color:#ffffff;">{MODEL_PATH}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# auto refresh (kecepatan default tidak diubah)
st_autorefresh(interval=2000, limit=None, key="refresh")

left, right = st.columns([1, 2])

# -----------------------
# LEFT PANEL (VISUAL UPGRADE)
# -----------------------
with left:
    # CONNECTION CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="stat">', unsafe_allow_html=True)
    connected = len(st.session_state.logs) > 0
    dot_color = "green" if connected else "red"
    st.markdown(f'<div class="dot {dot_color}"></div>', unsafe_allow_html=True)
    st.markdown(f'<div><p class="title">MQTT Broker</p><p class="sub">{MQTT_BROKER}:{MQTT_PORT}</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # LAST READING CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p style="margin:0; font-weight:700; font-size:16px;">Latest Reading</p>', unsafe_allow_html=True)
    if st.session_state.last:
        last = st.session_state.last
        # status icon for prediction
        pred = str(last.get("pred"))
        icon = "❔"
        color_label = "yellow"
        if pred:
            p = pred.lower()
            if "panas" in p or "hot" in p:
                icon = "🔥"; color_label = "red"
            elif "dingin" in p or "cold" in p:
                icon = "❄️"; color_label = "yellow"
            else:
                icon = "✅"; color_label = "green"

        st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<div><b>Time:</b> {last["ts"]}<br><b>Temp:</b> {last["temp"]} °C<br><b>Hum:</b> {last["hum"]} %</div>'
                    f'<div style="text-align:right;"><div style="font-size:28px">{icon}</div><div style="color:var(--muted); font-size:12px">Prediction</div>'
                    f'<div style="font-weight:700">{last["pred"]}</div></div></div>', unsafe_allow_html=True)
    else:
        st.info("Waiting for data...")
    st.markdown('</div>', unsafe_allow_html=True)

    # MANUAL CONTROL CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p style="margin:0; font-weight:700; font-size:16px;">Manual Control</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    if c1.button("ALERT ON"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_ON"})
    if c2.button("ALERT OFF"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_OFF"})
    st.markdown('</div>', unsafe_allow_html=True)

    # ANOMALY SETTINGS CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p style="margin:0; font-weight:700; font-size:16px;">Anomaly Settings</p>', unsafe_allow_html=True)
    st.session_state.anomaly_window = st.slider("Window Size", 5, 200, st.session_state.anomaly_window)
    st.markdown('</div>', unsafe_allow_html=True)

    # DOWNLOAD CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p style="margin:0; font-weight:700; font-size:16px;">Download Logs</p>', unsafe_allow_html=True)
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"iot_logs_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")
    else:
        st.write("No logs yet")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# RIGHT PANEL (VISUAL UPGRADE)
# -----------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p style="margin:0; font-weight:700; font-size:16px;">Live Chart — Last {MAX_POINTS} points</p>', unsafe_allow_html=True)
    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])

    if not df_plot.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_plot["ts"],
            y=df_plot["temp"],
            mode="lines+markers",
            name="Temperature (°C)",
            line=dict(width=2),
            marker=dict(size=6)
        ))

        # humidity trace
        fig.add_trace(go.Scatter(
            x=df_plot["ts"],
            y=df_plot["hum"],
            mode="lines+markers",
            name="Humidity (%)",
            yaxis="y2",
            line=dict(width=2, dash="dash"),
            marker=dict(size=6)
        ))

        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=10, b=10, l=10, r=10),
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p style="margin:0; font-weight:700; font-size:16px;">Recent Logs</p>', unsafe_allow_html=True)
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(50))
    else:
        st.write("—")
    st.markdown('</div>', unsafe_allow_html=True)
