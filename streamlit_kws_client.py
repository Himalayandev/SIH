#!/usr/bin/env python3
"""
=============================================================================
STREAMLIT KWS TCP EDGE CLIENT GUI
Modern Streamlit Dashboard & Client for Low-Latency Edge-Cloud System.
Features:
- Pre-warmed persistent TCP socket (0ms wake delay)
- Length-Prefixed TLV Framed Audio Chunks (0x02) & 0xFF Stream End
- 100ms Lookback Audio Pre-Roll Buffer (Zero command clipping)
- Real-time 18-Byte + UTF-8 STT Text Telemetry & Latency Parsing
- Streamlit Dark UI Dashboard with Live KWS Probabilities & Server Metrics
=============================================================================
"""

import os
import sys
import time
import socket
import threading
import struct
import numpy as np
import psutil
import streamlit as st

# Automatically bootstrap Streamlit if script is run directly via Python
if __name__ == "__main__":
    if not st.runtime.exists():
        import subprocess
        cmd = [sys.executable, "-m", "streamlit", "run", __file__]
        print(f"🚀 Bootstrapping Streamlit Dashboard (automatically selecting next free port)...")
        subprocess.run(cmd)
        sys.exit(0)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Configuration Defaults (Match KWS model)
MODEL_PATH = "res8_activate_int8.tflite"
CLASSES = ["background", "unknown", "activate"]
ACTIVATE_IDX = CLASSES.index("activate")

SAMPLE_RATE = 16000
DURATION = 1.00
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 640
HOP_LENGTH = 320
N_MELS = 40
N_FRAMES = 49

PROTOCOL_HEARTBEAT   = 0x00
PROTOCOL_SYN         = 0x01
PROTOCOL_SYN_ACK     = 0x06
PROTOCOL_SYN_DENIED  = 0x07
PROTOCOL_SYN_PENDING = 0x08
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END  = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Streamlit Page Configuration
st.set_page_config(
    page_title="KWS TCP Edge Client GUI",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for modern dark UI styling
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .log-box {
        background-color: #020617;
        color: #10b981;
        font-family: 'Courier New', monospace;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px;
        height: 200px;
        overflow-y: scroll;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared App State (Accessed by background thread and main Streamlit thread)
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.is_listening = False
        self.status = "Stopped"
        self.probs = [0.0, 0.0, 0.0]
        self.cpu_usage = 0.0
        self.sent_bytes = 0
        self.stream_rem = 0.0
        self.logs = []
        self.server_connected = False
        self.trigger_count = 0
        self.last_stt_text = ""
        self.last_transfer_ms = 0
        self.last_stt_ms = 0
        self.last_total_ms = 0
        self.lock = threading.Lock()

    def add_log(self, msg):
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {msg}")
            if len(self.logs) > 30:
                self.logs.pop(0)

@st.cache_resource
def get_app_state():
    return AppState()

state = get_app_state()

# ---------------------------------------------------------------------------
# Preprocessing Helper (Matches training pipeline)
# ---------------------------------------------------------------------------
def extract_kws_features(audio: np.ndarray, target_frames: int = 49) -> np.ndarray:
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_SAMPLES]

    if HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP_LENGTH, power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
    else:
        log_mel = np.zeros((N_MELS, target_frames), dtype=np.float32)

    n = log_mel.shape[1]
    if n >= target_frames:
        start = (n - target_frames) // 2
        mat = log_mel[:, start:start + target_frames]
    else:
        mat = np.pad(log_mel, ((0, 0), (0, target_frames - n)), mode="constant")

    return np.clip((mat + 80.0) / 80.0, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Background KWS Thread Engine
# ---------------------------------------------------------------------------
def kws_engine_loop(state_obj, host, port, act_thresh, vad_thresh, stream_duration, step_sec):
    state_obj.add_log("Loading local INT8 KWS model...")
    target_frames = 49
    if HAS_TF and os.path.exists(MODEL_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            in_details = interpreter.get_input_details()[0]
            in_idx = in_details['index']
            out_idx = interpreter.get_output_details()[0]['index']
            in_scale, in_zero_point = in_details['quantization']
            out_scale, out_zero_point = interpreter.get_output_details()[0]['quantization']

            # Dynamically detect required frame dimension from model input shape
            inp_shape = in_details['shape']
            if len(inp_shape) >= 3:
                target_frames = inp_shape[2] if inp_shape[2] != N_MELS else inp_shape[1]
            state_obj.add_log(f"Model loaded: input shape {list(inp_shape)}, target_frames={target_frames}")
        except Exception as e:
            interpreter = None
            state_obj.add_log(f"Model initialization warning: {e}")
    else:
        interpreter = None
        state_obj.add_log("⚠️ KWS model missing or TF not available. Running in manual/VAD trigger mode.")

    process = psutil.Process(os.getpid())
    last_cpu_time = 0.0

    audio_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    preroll_buffer = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.int16)  # 100ms lookback
    step_size = int(SAMPLE_RATE * step_sec)

    sock = None

    def connect_server():
        nonlocal sock
        state_obj.status = "Connecting"
        while state_obj.is_listening:
            state_obj.add_log(f"Connecting to server {host}:{port}...")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(3.0)
                s.connect((host, port))

                # Send Handshake SYN (0x01)
                try:
                    s.sendall(bytes([PROTOCOL_SYN]))
                    ack = s.recv(1)
                    if ack:
                        state_obj.add_log(f"Handshake ACK received: {hex(ack[0])}")
                except Exception as ex:
                    state_obj.add_log(f"Handshake notice: {ex}")

                s.settimeout(None)
                sock = s
                with state_obj.lock:
                    state_obj.server_connected = True
                state_obj.add_log(f"🟢 Connected to STT Server {host}:{port}!")
                state_obj.status = "Listening"
                break

            except Exception as e:
                with state_obj.lock:
                    state_obj.server_connected = False
                state_obj.add_log(f"❌ Connection failed to {host}:{port}: {e}")
                for _ in range(10):
                    if not state_obj.is_listening:
                        break
                    time.sleep(0.2)

    connect_server()

    try:
        while state_obj.is_listening:
            if not HAS_SOUNDDEVICE:
                state_obj.add_log("❌ sounddevice module not available!")
                break

            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16', blocksize=step_size) as mic:
                while state_obj.is_listening:
                    pcm_data, _ = mic.read(step_size)
                    raw_pcm16 = pcm_data[:, 0]
                    
                    # Pre-roll buffer update
                    n_samp = len(raw_pcm16)
                    if n_samp < len(preroll_buffer):
                        preroll_buffer = np.roll(preroll_buffer, -n_samp)
                        preroll_buffer[-n_samp:] = raw_pcm16
                    else:
                        preroll_buffer = raw_pcm16[-len(preroll_buffer):]

                    # Audio buffer update
                    samples = raw_pcm16.astype(np.float32) / 32768.0
                    if n_samp < len(audio_buffer):
                        audio_buffer = np.roll(audio_buffer, -n_samp)
                        audio_buffer[-n_samp:] = samples
                    else:
                        audio_buffer = samples[-len(audio_buffer):]

                    # CPU tracking
                    current_time = time.time()
                    if current_time - last_cpu_time >= 1.0:
                        state_obj.cpu_usage = process.cpu_percent() / max(1, psutil.cpu_count())
                        last_cpu_time = current_time

                    # Voice Activity Check
                    rms = np.sqrt(np.mean(audio_buffer**2))
                    if rms < vad_thresh:
                        with state_obj.lock:
                            state_obj.probs = [1.0, 0.0, 0.0]
                        continue

                    # KWS Inference
                    if interpreter:
                        feat = extract_kws_features(audio_buffer, target_frames=target_frames)
                        inp = feat[np.newaxis, ..., np.newaxis].astype(np.float32)

                        if in_scale > 0:
                            inp = np.round(inp / in_scale + in_zero_point).astype(np.int8)

                        interpreter.set_tensor(in_idx, inp)
                        interpreter.invoke()
                        out = interpreter.get_tensor(out_idx)

                        probs = (out.astype(np.float32) - out_zero_point) * out_scale if out_scale > 0 else out
                        act_prob = probs[0][ACTIVATE_IDX]
                        unk_prob = probs[0][CLASSES.index("unknown")]
                        bg_prob = probs[0][CLASSES.index("background")]
                    else:
                        act_prob = 0.95 if rms >= vad_thresh else 0.0
                        unk_prob = 0.0
                        bg_prob = 1.0 - act_prob

                    with state_obj.lock:
                        state_obj.probs = [float(bg_prob), float(unk_prob), float(act_prob)]

                    # Check for Trigger
                    if act_prob >= act_thresh:
                        with state_obj.lock:
                            state_obj.trigger_count += 1
                        state_obj.add_log(f"⚡ Triggered KWS: 'Activate' ({act_prob*100:.1f}%) | Triggers: {state_obj.trigger_count}")
                        if not state_obj.server_connected:
                            state_obj.add_log("⚠️ Cannot stream: Server disconnected. Skipping...")
                            audio_buffer.fill(0)
                            time.sleep(1.0)
                            continue
                        break
            
            # Streaming flow
            if state_obj.is_listening and act_prob >= act_thresh and state_obj.server_connected and sock:
                state_obj.status = "Streaming"
                state_obj.add_log("🎙️ Streaming audio chunks over persistent TCP socket...")
                
                try:
                    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
                    start_time = time.time()
                    state_obj.sent_bytes = 0
                    
                    # 1. Send Pre-roll 100ms buffer first
                    preroll_bytes = preroll_buffer.tobytes()
                    header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(preroll_bytes))
                    sock.sendall(header + preroll_bytes)
                    state_obj.sent_bytes += len(preroll_bytes)

                    # 2. Stream live audio
                    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16', blocksize=chunk_size) as stream_mic:
                        while state_obj.is_listening and (time.time() - start_time < stream_duration):
                            pcm_chunk, _ = stream_mic.read(chunk_size)
                            chunk_bytes = pcm_chunk.tobytes()
                            
                            header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(chunk_bytes))
                            sock.sendall(header + chunk_bytes)
                            
                            with state_obj.lock:
                                state_obj.sent_bytes += len(chunk_bytes)
                                state_obj.stream_rem = max(0.0, stream_duration - (time.time() - start_time))

                            # Query CPU
                            current_time = time.time()
                            if current_time - last_cpu_time >= 1.0:
                                state_obj.cpu_usage = process.cpu_percent() / max(1, psutil.cpu_count())
                                last_cpu_time = current_time

                    # 3. Send Stream End Frame (0xFF)
                    end_header = struct.pack("<BH", PROTOCOL_STREAM_END, 0)
                    sock.sendall(end_header)
                    state_obj.add_log("--> Sent Stream End (0xFF). Awaiting server Transit ACK & Telemetry...")

                    # 4. Read Transit ACK (0x7F)
                    transit_ack = sock.recv(1)
                    if transit_ack and transit_ack[0] == PROTOCOL_TRANSIT_ACK:
                        state_obj.add_log("⚡ [TRANSIT ACK] Received 0x7F from server!")

                    # 5. Read 18-Byte Telemetry Header + UTF-8 STT Output
                    telemetry = sock.recv(18)
                    if len(telemetry) == 18:
                        audio_dur, edge_ms, transfer_ms, asr_ms, text_len = struct.unpack("<IIIIH", telemetry)
                        text_str = ""
                        if text_len > 0:
                            text_bytes = sock.recv(text_len)
                            text_str = text_bytes.decode('utf-8', errors='ignore')

                        with state_obj.lock:
                            state_obj.last_stt_text = text_str
                            state_obj.last_transfer_ms = transfer_ms
                            state_obj.last_stt_ms = asr_ms
                            state_obj.last_total_ms = transfer_ms + asr_ms

                        state_obj.add_log(f"📊 STT Output: \"{text_str}\" | Transfer: {transfer_ms}ms | STT Compute: {asr_ms}ms")

                    # Reconnect socket for next stream
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None
                    with state_obj.lock:
                        state_obj.server_connected = False

                except Exception as e:
                    state_obj.add_log(f"⚠️ Connection error during stream: {e}")
                    with state_obj.lock:
                        state_obj.server_connected = False
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None
                
                # Reset to Listening state and reconnect
                if state_obj.is_listening:
                    connect_server()
                    audio_buffer.fill(0)
                    time.sleep(0.5)

    except Exception as e:
        state_obj.status = "Error"
        state_obj.add_log(f"Engine crash: {e}")
    finally:
        state_obj.status = "Stopped"
        state_obj.probs = [0.0, 0.0, 0.0]
        state_obj.cpu_usage = 0.0
        with state_obj.lock:
            state_obj.server_connected = False
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        state_obj.add_log("KWS listening engine stopped cleanly.")


# ---------------------------------------------------------------------------
# Streamlit UI Rendering
# ---------------------------------------------------------------------------
st.title("🎙️ KWS TCP Edge Client Dashboard")
st.markdown("Monitor wake-word detection probabilities, process CPU load, network streaming metrics, and real-time STT results.")

# Sidebar Configurations
st.sidebar.header("⚙️ Client Configurations")
host = st.sidebar.text_input("Server Host", value="192.168.137.1", disabled=state.is_listening)
port = st.sidebar.number_input("Server Port", min_value=1, max_value=65535, value=8088, disabled=state.is_listening)

st.sidebar.subheader("🔍 Thresholds & Durations")
act_thresh = st.sidebar.slider("Activation Probability Threshold", 0.10, 0.99, 0.70, 0.05, disabled=state.is_listening)
vad_thresh = st.sidebar.slider("VAD RMS Noise Floor Threshold", 0.001, 0.050, 0.005, 0.001, format="%.3f", disabled=state.is_listening)
stream_duration = st.sidebar.slider("Speech Streaming Duration (s)", 1.0, 10.0, 3.0, 0.5, disabled=state.is_listening)
step_sec = st.sidebar.slider("KWS Window Step Size (s)", 0.05, 0.50, 0.15, 0.05, disabled=state.is_listening)

st.sidebar.markdown("---")

# Toggle listening loop
if not state.is_listening:
    if st.sidebar.button("🟢 Start Listening Engine", use_container_width=True):
        state.is_listening = True
        state.status = "Starting"
        state.trigger_count = 0
        state.logs = []
        state.last_stt_text = ""
        
        # Start background loop in thread
        engine_thread = threading.Thread(
            target=kws_engine_loop,
            args=(state, host, port, act_thresh, vad_thresh, stream_duration, step_sec),
            daemon=True
        )
        engine_thread.start()
        st.rerun()
else:
    if st.sidebar.button("🛑 Stop Listening Engine", use_container_width=True):
        state.is_listening = False
        st.rerun()

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live KWS Probabilities")
    
    # State color mapping
    status_colors = {
        "Stopped": "#64748b",
        "Starting": "#eab308",
        "Connecting": "#eab308",
        "Listening": "#22c55e",
        "Streaming": "#f97316",
        "Error": "#ef4444"
    }
    current_status = state.status
    color = status_colors.get(current_status, "#64748b")
    
    st.markdown(f"""
        <div style="background-color: {color}22; border-left: 6px solid {color}; padding: 12px; border-radius: 4px; margin-bottom: 20px;">
            <span style="font-weight: bold; color: {color}; font-size: 1.2rem;">STATUS: {current_status.upper()}</span>
        </div>
    """, unsafe_allow_html=True)

    # Render Bar chart of probabilities
    probs = state.probs
    chart_data = {
        "Classes": CLASSES,
        "Probability (%)": [p * 100 for p in probs]
    }
    st.bar_chart(
        data=chart_data,
        x="Classes",
        y="Probability (%)",
        color="#38bdf8"
    )

    # Render Real-time Speech-To-Text (STT) Result Card
    st.subheader("🗣️ Server Speech-To-Text (STT) Result")
    if state.last_stt_text:
        st.markdown(f"""
            <div style="background-color: #1e293b; border: 2px solid #00e5ff; border-radius: 8px; padding: 1.2rem; margin-top: 10px;">
                <span style="color: #94a3b8; font-size: 0.9rem; font-weight: bold;">LAST TRANSCRIBED TEXT</span>
                <h2 style="margin: 5px 0; color: #00e5ff; font-size: 1.8rem;">"{state.last_stt_text}"</h2>
                <div style="display: flex; gap: 15px; margin-top: 10px; color: #e2e8f0; font-size: 0.9rem;">
                    <span>⏱️ Transfer: <b>{state.last_transfer_ms} ms</b></span>
                    <span>⚡ STT Compute: <b>{state.last_stt_ms} ms</b></span>
                    <span>🚀 Total Latency: <b>{state.last_total_ms} ms</b></span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting wake-word activation and server STT response...")

with col2:
    st.subheader("🖥️ Client Metrics")
    
    # Server Connection Status Indicator
    if state.is_listening:
        if state.server_connected:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 6px solid #22c55e;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">CONNECTION STATUS</span>
                    <h3 style="margin: 0; color: #22c55e; font-size: 1.3rem;">🟢 Server Connected</h3>
                    <span style="color: #64748b; font-size: 0.8rem;">Active link to {host}:{port}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 6px solid #ef4444;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">CONNECTION STATUS</span>
                    <h3 style="margin: 0; color: #ef4444; font-size: 1.3rem;">🔴 Reconnecting...</h3>
                    <span style="color: #64748b; font-size: 0.8rem;">Searching for {host}:{port}...</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="metric-card" style="border-left: 6px solid #64748b;">
                <span style="color: #94a3b8; font-size: 0.9rem;">CONNECTION STATUS</span>
                <h3 style="margin: 0; color: #64748b; font-size: 1.3rem;">⚪ Offline</h3>
                <span style="color: #64748b; font-size: 0.8rem;">Listening engine stopped</span>
            </div>
        """, unsafe_allow_html=True)
        
    # Trigger Count Metric
    st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid #38bdf8;">
            <span style="color: #94a3b8; font-size: 0.9rem;">TOTAL WAKE-WORD TRIGGERS</span>
            <h2 style="margin: 0; font-size: 2.2rem; color: #38bdf8;">{state.trigger_count}</h2>
            <span style="color: #64748b; font-size: 0.8rem;">Session triggers count</span>
        </div>
    """, unsafe_allow_html=True)
        
    # CPU Metric
    st.markdown(f"""
        <div class="metric-card">
            <span style="color: #94a3b8; font-size: 0.9rem;">AVERAGE CPU USAGE</span>
            <h2 style="margin: 0; font-size: 2.2rem; color: #38bdf8;">{state.cpu_usage:.1f}%</h2>
            <span style="color: #64748b; font-size: 0.8rem;">Normalized by CPU Core Count</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Network Streaming Metrics
    if current_status == "Streaming":
        st.markdown(f"""
            <div class="metric-card" style="border-color: #f97316;">
                <span style="color: #94a3b8; font-size: 0.9rem;">STREAM TRANSMISSION</span>
                <h2 style="margin: 0; font-size: 2.2rem; color: #f97316;">{state.stream_rem:.1f}s remaining</h2>
                <span style="color: #94a3b8; font-size: 0.85rem;">Bytes Sent: <b>{state.sent_bytes}</b> bytes</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="metric-card">
                <span style="color: #64748b; font-size: 0.9rem;">STREAM TRANSMISSION</span>
                <h2 style="margin: 0; font-size: 2.2rem; color: #334155;">Idle</h2>
                <span style="color: #64748b; font-size: 0.8rem;">Awaiting Wake-Word Trigger...</span>
            </div>
        """, unsafe_allow_html=True)

# Console Logs Display
st.subheader("📋 Console Logs")
logs_content = "\n".join(reversed(state.logs)) if state.logs else "Awaiting engine startup..."
st.markdown(f'<div class="log-box">{logs_content}</div>', unsafe_allow_html=True)

# Periodic update cycle for Streamlit rerun
if state.is_listening:
    time.sleep(0.1)  # Refresh dashboard at ~10Hz
    st.rerun()
