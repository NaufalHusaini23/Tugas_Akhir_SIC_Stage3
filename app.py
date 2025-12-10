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
# ULTRA AURORA UI (ANIMATED) - ALL VISUALS ONLY
# =====================================================
st.set_page_config(page_title="IoT ML — Ultra Aurora Dashboard", layout="wide")

st.markdown(
"""
<style>
/* === Full-page animated aurora background === */
@keyframes aurora {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
body, .css-18e3th9 { 
  background: radial-gradient(circle at 10% 10%, rgba(20,18,40,0.9), rgba(5,7,20,0.95) 40%), linear-gradient(90deg, rgba(40,10,60,0.35), rgba(10,20,60,0.35));
  background-blend-mode: screen, overlay;
  color: #e6eef8;
}

/* animated aurora layer */
.aurora {
  position: fixed;
  top: -20%;
  left: -20%;
  width: 140vw;
  height: 60vh;
  z-index: 0;
  background: linear-gradient(120deg, rgba(90,45,255,0.18), rgba(30,200,255,0.12), rgba(120,60,255,0.14));
  filter: blur(60px) saturate(140%);
  transform: rotate(-6deg);
  animation: aurora 18s ease-in-out infinite;
  opacity: 0.95;
  pointer-events: none;
}

/* subtle second aurora band */
.aurora2 {
  position: fixed;
  top: 40%;
  left: -10%;
  width: 120vw;
  height: 45vh;
  z-index: 0;
  background: linear-gradient(60deg, rgba(255,100,220,0.06), rgba(60,120,255,0.06));
  filter: blur(90px) saturate(120%);
  animation: aurora 24s ease-in-out infinite reverse;
  pointer-events: none;
}

/* glass card base */
.glass {
  position: relative;
  z-index: 2;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 16px;
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 8px 30px rgba(0,0,0,0.6);
  backdrop-filter: blur(8px) saturate(120%);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}

/* floating effect on hover */
.glass:hover {
  transform: translateY(-6px);
  box-shadow: 0 18px 50px rgba(20,20,60,0.7);
}

/* neon border animation */
.neon {
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 0 18px rgba(80,60,255,0.06), inset 0 1px 0 rgba(255,255,255,0.02);
  position: relative;
}
.neon:after {
  content: "";
  position: absolute;
  inset: -2px;
  border-radius: 16px;
  background: linear-gradient(90deg, rgba(93,43,255,0.12), rgba(55,150,255,0.10));
  filter: blur(18px);
  z-index: -1;
  opacity: 0.9;
}

/* header style */
.header {
  padding: 18px 22px;
  margin-bottom: 12px;
  border-radius: 12px;
  position: relative;
}

/* title */
.h1 {
  font-size: 26px;
  color: #e9f0ff;
  margin: 0 0 6px 0;
  font-weight: 700;
  letter-spacing: -0.2px;
}
.h2 {
  margin: 0;
  color: #bfc8ff;
  font-size: 13px;
}

/* status dot */
.status-dot {
  width: 12px; height: 12px; border-radius: 50%;
  display:inline-block; margin-right:8px;
  box-shadow: 0 6px 18px rgba(70,60,255,0.12);
}
.dot-green { background: #34d399; }
.dot-yellow { background: #fbbf24; }
.dot-red { background: #fb7185; }

/* buttons neon animated */
.stButton>button {
  background: linear-gradient(90deg,#6c5cff 0%,#3dd1ff 100%);
  color:#051025;
  border-radius:10px;
  padding:10px 14px;
  font-weight:700;
  box-shadow: 0 8px 30px rgba(80,60,255,0.18);
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton>button:hover { transform: translateY(-3px); box-shadow:0 16px 40px rgba(80,60,255,0.26); }

/* subtle pulsing for ALERT ON button when active (class added inline) */
.pulse {
  animation: pulse 1.8s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 8px 30px rgba(80,60,255,0.18); }
  50% { box-shadow: 0 20px 48px rgba(80,60,255,0.28); transform: translateY(-2px); }
  100% { box-shadow: 0 8px 30px rgba(80,60,255,0.18); transform: translateY(0); }
}

/* download button */
.stDownloadButton>button { background: linear-gradient(90deg,#3b5cff,#7b38ff); color:white; border-radius:10px; }

/* small badges for prediction */
.badge {
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-weight:700;
  color:#001020;
  margin-left:6px;
  font-size:13px;
}
.badge-hot { background: linear-gradient(90deg,#ff7a7a,#ff3b3b); box-shadow:0 8px 20px rgba(255,60,60,0.12); color:white; }
.badge-cold { background: linear-gradient(90deg,#7ad8ff,#3bb6ff); box-shadow:0 8px 20px rgba(60,180,255,0.12); color:white; }
.badge-normal { background: linear-gradient(90deg,#b7f7c6,#3df6a0); box-shadow:0 8px 20px rgba(60,200,140,0.08); color:#002017; }

/* table style */
.stDataFrame table { background: transparent; color:#dfe8ff; }

/* plotly overrides */
.plotly-graph-div { border-radius:12px; overflow:hidden; }

/* small responsive tweaks */
@media (max-width: 800px) {
  .aurora { display:none; }
  .aurora2 { display:none; }
}
</style>

<div class="aurora"></div>
<div class="aurora2"></div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# CONFIG (do not change behavior)
# =====================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/sic/make/sensor"
TOPIC_OUTPUT = "iot/sic/make/output"
MODEL_PATH = "iot_temp_model.pkl"

MAX_POINTS = 200
ANOMALY_Z_THRESHOLD = 3.0

# =====================================================
# SESSION STATE (do not modify)
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
# LOAD MODEL (same)
# =====================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model '{path}': {e}")
        return None

model = load_model(MODEL_PATH)
st.session_state.model_loaded = model is not None

# =====================================================
# MQTT WORKER (same)
# =====================================================
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

# =====================================================
# PROCESS INCOMING (same logic)
# =====================================================
def process_incoming():
    updated = False
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

        # SEND ML RESULT BACK TO ESP32 (same)
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

# =====================================================
# UI (structure & functions unchanged; visuals upgraded)
# =====================================================
st.markdown("""
<div class="header glass neon" style="display:flex; justify-content:space-between; align-items:center;">
  <div>
    <div class="h1">🌌 IoT ML Realtime — Pemantauan Ruang Server</div>
    <div class="h2">Smart Environment Monitoring • Streamlit + MQTT + scikit-learn</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:12px; color:#cbd6ff">Model:</div>
    <div style="font-weight:700; color:#ffffff">{}</div>
  </div>
</div>
""".format(MODEL_PATH), unsafe_allow_html=True)

# auto-refresh remains same
st_autorefresh(interval=2000, limit=None, key="ultra-refresh")

left, right = st.columns([1, 2])

# LEFT PANEL (visual)
with left:
    st.markdown('<div class="glass neon">', unsafe_allow_html=True)
    st.markdown('### 📡 Connection Status')
    connected = len(st.session_state.logs) > 0
    dot = 'dot-green' if connected else 'dot-red'
    st.markdown(f'<div style="display:flex; align-items:center; gap:10px;"><span class="status-dot { "dot-green" if connected else "dot-red" }"></span><div style="font-weight:700">{MQTT_BROKER}:{MQTT_PORT}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### 📊 Latest Reading')
    if st.session_state.last:
        last = st.session_state.last
        # choose badge
        pred = str(last.get("pred")) if last.get("pred") else ""
        badge_html = '<span class="badge badge-hot">🔥 PANAS</span>'
        if pred:
            p = pred.lower()
            if "panas" in p or "hot" in p:
                badge_html = '<span class="badge badge-hot">🔥 PANAS</span>'
            elif "dingin" in p or "cold" in p:
                badge_html = '<span class="badge badge-cold">❄️ DINGIN</span>'
            else:
                badge_html = '<span class="badge badge-normal">✅ NORMAL</span>'
        st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<div><b>Time:</b> {last["ts"]}<br><b>Temp:</b> {last["temp"]} °C<br><b>Hum:</b> {last["hum"]} %</div>'
                    f'<div style="text-align:right;">{badge_html}<div style="color:#98a6ff; font-size:12px; margin-top:6px">Confidence: {last.get("conf")}</div></div>'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.info("Waiting for data...")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### 🔧 Manual Control')
    c1, c2 = st.columns([1,1])
    if c1.button("ALERT ON"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_ON"})
    if c2.button("ALERT OFF"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_OFF"})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### ⚙️ Anomaly Settings')
    st.session_state.anomaly_window = st.slider("Window Size", 5, 200, st.session_state.anomaly_window)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### 📥 Download Logs')
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, file_name=f"iot_logs_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")
    else:
        st.write("No logs yet")
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT PANEL (visual)
with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f'### 📈 Live Chart (Last {MAX_POINTS} points)')
    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])

    if not df_plot.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["temp"],
            mode="lines+markers", name="Temperature (°C)",
            line=dict(width=2, color='#ff9f43'),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["hum"],
            mode="lines+markers", name="Humidity (%)",
            line=dict(width=2, dash='dash', color='#3bd1ff'),
            marker=dict(size=6),
            yaxis='y2'
        ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(12,12,20,0.3)',
            yaxis=dict(title='Temperature (°C)'),
            yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=10,b=6,l=6,r=6),
            height=460
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Ensure ESP32 publishes to the sensor topic.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### 📝 Recent Logs')
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(50))
    else:
        st.write("—")
    st.markdown('</div>', unsafe_allow_html=True)

# final process incoming call (keeps behavior)
process_incoming()

