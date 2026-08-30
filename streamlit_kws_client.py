#!/usr/bin/env python3
"""
=============================================================================
STREAMLIT KWS TCP EDGE CLIENT GUI
Modern Streamlit Dashboard & Client for Low-Latency Edge-Cloud System.
- Pre-warmed persistent TCP socket (0ms wake delay)
- Length-Prefixed TLV Framed Audio Chunks (0x02, no 0xFF collisions)
- 100ms Lookback Audio Pre-Roll Buffer (Zero command clipping)
- 18-Byte + UTF-8 STT Text Telemetry Parsing
- Idle TCP Keepalive Heartbeat (0x00)
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
        self.last_asr_ms = 0
        self.last_rtt_ms = 0.0
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
# Preprocessing Helper (Zero-dependency numpy fallback if librosa missing)
# ---------------------------------------------------------------------------
def extract_kws_features(audio: np.ndarray) -> np.ndarray:
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
        frames = []
        for i in range(0, len(audio) - N_FFT + 1, HOP_LENGTH):
            windowed = audio[i:i + N_FFT] * np.hanning(N_FFT)
            fft_mag = np.abs(np.fft.rfft(windowed)) ** 2
            frames.append(fft_mag)
        if not frames:
            frames = [np.zeros(N_FFT // 2 + 1, dtype=np.float32)]
        stft = np.column_stack(frames)
        n_freqs = stft.shape[0]
        mel_filters = np.linspace(0, n_freqs - 1, N_MELS + 2, dtype=int)
        fb = np.zeros((N_MELS, n_freqs))
        for m in range(N_MELS):
            fb[m, mel_filters[m]:mel_filters[m+2]] = 1.0
        mel = np.dot(fb, stft)
        log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))

    n = log_mel.shape[1]
    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        mat = log_mel[:, start:start + N_FRAMES]
    else:
        mat = np.pad(log_mel, ((0, 0), (0, N_FRAMES - n)), mode="constant")

    return np.clip((mat + 80.0) / 80.0, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Background KWS Thread Engine (With Pre-Warmed Sockets & TLV Framing)
# ---------------------------------------------------------------------------
def kws_engine_loop(state_obj, host, port, act_thresh, vad_thresh, stream_duration, step_sec):
    state_obj.add_log("Initializing INT8 KWS Model & Pre-Warmed Protocol Engine...")

    interpreter = None
    if HAS_TF and os.path.exists(MODEL_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            in_idx = interpreter.get_input_details()[0]['index']
            out_idx = interpreter.get_output_details()[0]['index']
            in_scale, in_zero_point = interpreter.get_input_details()[0]['quantization']
            out_scale, out_zero_point = interpreter.get_output_details()[0]['quantization']
            state_obj.add_log(f"✅ TFLite Model '{MODEL_PATH}' loaded successfully.")
        except Exception as e:
            state_obj.add_log(f"⚠️ Model load warning: {e}")
    else:
        state_obj.add_log("⚠️ TFLite model or TensorFlow unavailable. Running in Manual Trigger mode.")

    process = psutil.Process(os.getpid())
    last_cpu_time = 0.0

    audio_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    step_size = int(SAMPLE_RATE * step_sec)

    # 100ms Pre-Roll Lookback Ring Buffer (3 x 100ms frames)
    preroll_ring = []
    
    sock = None

    def connect_and_prewarm():
        nonlocal sock
        state_obj.status = "Connecting"
        while state_obj.is_listening:
            state_obj.add_log(f"Pre-warming TCP socket connection to {host}:{port}...")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(5.0)
                s.connect((host, port))
                
                # Send SYN Handshake (0x01)
                s.sendall(bytes([PROTOCOL_SYN]))
                ack = s.recv(1)
                if ack and ack[0] == PROTOCOL_SYN_ACK:
                    sock = s
                    with state_obj.lock:
                        state_obj.server_connected = True
                    state_obj.add_log("🤝 [PRE-WARMED SOCKET READY] 0x01 -> 0x06 SYN-ACK Verified! (0ms wake delay)")
                    state_obj.status = "Listening"
                    return True
                else:
                    s.close()
            except Exception as e:
                with state_obj.lock:
                    state_obj.server_connected = False
                state_obj.add_log(f"❌ Connection failed: {e}. Retrying in 2 seconds...")
                for _ in range(20):
                    if not state_obj.is_listening:
                        break
                    time.sleep(0.1)
        return False

    # Connect persistent socket on boot
    connect_and_prewarm()
    last_heartbeat_time = time.time()

    try:
        while state_obj.is_listening:
            if not state_obj.server_connected or sock is None:
                if not connect_and_prewarm():
                    continue

            # Idle TCP Keepalive Heartbeat (0x00) every 20s
            if time.time() - last_heartbeat_time > 20.0:
                try:
                    sock.sendall(bytes([PROTOCOL_HEARTBEAT]))
                    last_heartbeat_time = time.time()
                except Exception:
                    state_obj.add_log("⚠️ Heartbeat ping failed. Socket re-establishing...")
                    state_obj.server_connected = False
                    continue

            # Audio Input & KWS Loop
            if HAS_SOUNDDEVICE:
                try:
                    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16', blocksize=step_size) as mic:
                        while state_obj.is_listening and state_obj.server_connected:
                            pcm_data, _ = mic.read(step_size)
                            pcm_bytes = pcm_data.tobytes()

                            # Maintain 100ms Pre-Roll Lookback Ring Buffer
                            preroll_ring.append(pcm_bytes)
                            if len(preroll_ring) > 3:
                                preroll_ring.pop(0)

                            samples = pcm_data[:, 0].astype(np.float32) / 32768.0
                            audio_buffer = np.roll(audio_buffer, -step_size)
                            audio_buffer[-step_size:] = samples

                            current_time = time.time()
                            if current_time - last_cpu_time >= 1.0:
                                state_obj.cpu_usage = process.cpu_percent() / psutil.cpu_count()
                                last_cpu_time = current_time

                            rms = np.sqrt(np.mean(audio_buffer**2))
                            if rms < vad_thresh:
                                with state_obj.lock:
                                    state_obj.probs = [1.0, 0.0, 0.0]
                                continue

                            act_prob = 0.0
                            if interpreter:
                                feat = extract_kws_features(audio_buffer)
                                inp = feat[np.newaxis, ..., np.newaxis].astype(np.float32)
                                if in_scale > 0:
                                    inp = np.round(inp / in_scale + in_zero_point).astype(np.int8)

                                interpreter.set_tensor(in_idx, inp)
                                interpreter.invoke()
                                out = interpreter.get_tensor(out_idx)

                                probs = (out.astype(np.float32) - out_zero_point) * out_scale if out_scale > 0 else out
                                act_prob = float(probs[0][ACTIVATE_IDX])
                                unk_prob = float(probs[0][CLASSES.index("unknown")])
                                bg_prob = float(probs[0][CLASSES.index("background")])

                                with state_obj.lock:
                                    state_obj.probs = [bg_prob, unk_prob, act_prob]

                            if act_prob >= act_thresh:
                                with state_obj.lock:
                                    state_obj.trigger_count += 1
                                state_obj.add_log(f"⚡ 'ACTIVATE' Triggered ({act_prob*100:.1f}%)! Flushing 100ms Pre-Roll & Streaming...")
                                break

                            time.sleep(0.02)
                except Exception as e:
                    state_obj.add_log(f"⚠️ Sounddevice audio error: {e}")
                    time.sleep(1.0)
                    continue
            else:
                time.sleep(0.2)
                continue

            # Streaming Voice Stream over Pre-Warmed Socket using TLV Framing
            if state_obj.is_listening and state_obj.server_connected:
                state_obj.status = "Streaming"
                
                try:
                    # 1. Flush historical 100ms pre-roll lookback chunks first
                    for pr_chunk in preroll_ring:
                        header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(pr_chunk))
                        sock.sendall(header + pr_chunk)

                    chunk_size = int(SAMPLE_RATE * 0.1 * 2) # 100ms
                    start_time = time.time()
                    state_obj.sent_bytes = sum(len(c) for c in preroll_ring)
                    
                    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16', blocksize=int(SAMPLE_RATE * 0.1)) as mic:
                        while state_obj.is_listening and (time.time() - start_time < stream_duration):
                            pcm_data, _ = mic.read(int(SAMPLE_RATE * 0.1))
                            audio_bytes = pcm_data.tobytes()
                            
                            # Send Length-Prefixed TLV Frame (0x02)
                            header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(audio_bytes))
                            sock.sendall(header + audio_bytes)
                            
                            with state_obj.lock:
                                state_obj.sent_bytes += len(audio_bytes)
                                state_obj.stream_rem = max(0.0, stream_duration - (time.time() - start_time))

                    # 2. Send Stream End TLV Frame (0xFF)
                    t_end_sent = time.time()
                    end_header = struct.pack("<BH", PROTOCOL_STREAM_END, 0)
                    sock.sendall(end_header)
                    state_obj.add_log("--> Sent Stream End (0xFF). Waiting for Transit ACK (0x7F)...")

                    # 3. Read Instant Hardware Transit ACK (0x7F)
                    t_ack = sock.recv(1)
                    t_ack_received = time.time()
                    rtt_ms = (t_ack_received - t_end_sent) * 1000.0

                    if t_ack and t_ack[0] == PROTOCOL_TRANSIT_ACK:
                        state_obj.add_log(f"⚡ [TRANSIT ACK 0x7F] Received in {rtt_ms:.2f} ms!")

                    # 4. Read 18-Byte Telemetry Header + UTF-8 Transcribed Text String Payload
                    telemetry = sock.recv(18)
                    if len(telemetry) == 18:
                        audio_dur, edge_ms, net_ms, asr_ms, text_len = struct.unpack("<IIIIH", telemetry)
                        text_str = ""
                        if text_len > 0:
                            text_bytes = sock.recv(text_len)
                            text_str = text_bytes.decode('utf-8', errors='ignore')

                        with state_obj.lock:
                            state_obj.last_stt_text = text_str
                            state_obj.last_asr_ms = asr_ms
                            state_obj.last_rtt_ms = rtt_ms

                        state_obj.add_log(f"✅ [STT RESULT] Audio: {audio_dur}ms | RTT: {rtt_ms:.2f}ms | Server ASR Compute: {asr_ms}ms | STT: \"{text_str}\"")
                    
                    state_obj.status = "Listening"
                    audio_buffer.fill(0)
                    preroll_ring = []
                    last_heartbeat_time = time.time()
                    time.sleep(0.5)

                except Exception as e:
                    state_obj.add_log(f"⚠️ Connection error during streaming: {e}")
                    with state_obj.lock:
                        state_obj.server_connected = False
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None

    except Exception as e:
        state_obj.status = "Error"
        state_obj.add_log(f"Engine exception: {e}")
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
st.markdown("Monitor wake-word detection probabilities, process CPU load, and real-time STT telemetry in real-time.")

# Sidebar Configurations
st.sidebar.header("⚙️ Client Configurations")
host = st.sidebar.text_input("Server Host", value="127.0.0.1")
port = st.sidebar.number_input("Server Port", min_value=1, max_value=65535, value=8088)

st.sidebar.subheader("🔍 Thresholds & Durations")
act_thresh = st.sidebar.slider("Activation Probability Threshold", 0.10, 0.99, 0.65, 0.05)
vad_thresh = st.sidebar.slider("VAD RMS Noise Floor Threshold", 0.001, 0.050, 0.005, 0.001, format="%.3f")
stream_duration = st.sidebar.slider("Speech Streaming Duration (s)", 1.0, 10.0, 3.0, 0.5)
step_sec = st.sidebar.slider("KWS Window Step Size (s)", 0.05, 0.50, 0.15, 0.05)

st.sidebar.markdown("---")

# Toggle listening loop
if not state.is_listening:
    if st.sidebar.button("🟢 Start Listening Engine", use_container_width=True):
        state.is_listening = True
        state.status = "Starting"
        state.trigger_count = 0
        state.logs = []
        
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

    # Live Transcribed STT Card for Judges
    st.subheader("🗣️ Live Transcribed Speech Output (STT)")
    st_text = state.last_stt_text if state.last_stt_text else "(waiting for speech...)"
    st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid #a855f7;">
            <span style="color: #94a3b8; font-size: 0.9rem;">RECOGNIZED SPEECH TEXT</span>
            <h2 style="margin: 5px 0; font-size: 1.8rem; color: #a855f7;">"{st_text}"</h2>
            <span style="color: #64748b; font-size: 0.85rem;">Server ASR Compute: <b>{state.last_asr_ms} ms</b> | Net RTT: <b>{state.last_rtt_ms:.2f} ms</b></span>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("🖥️ Client Metrics")
    
    # Server Connection Status Indicator
    if state.is_listening:
        if state.server_connected:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 6px solid #22c55e;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">CONNECTION STATUS</span>
                    <h3 style="margin: 0; color: #22c55e; font-size: 1.3rem;">🟢 Pre-Warmed Socket Ready</h3>
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
