#!/usr/bin/env python3
"""
=============================================================================
NEW LOW-LATENCY ASR STT SERVER WITH REAL-TIME GUI DASHBOARD
Filename: new_server.py
Features:
 - Real-Time Dark Theme GUI (Tkinter + Custom TTK)
 - Client Device Connection Tracking (IP, Status, Connect Duration, Stream Count)
 - Precision Latency Measurement:
     * Data Transfer Latency (Client -> Server audio streaming time in ms)
     * STT Processing Latency (Faster-Whisper INT8 inference time in ms)
     * Total End-to-End Latency (ms)
 - Faster-Whisper CPU INT8 Engine with startup Warm-up (Zero Cold Start)
 - Dual Protocol Support: Custom TLV Binary TCP (ESP32/Edge Client) & Raw PCM
 - Built-in Test Client Simulator for instant latency verification
=============================================================================
"""

import os
import sys
import json
import time
import socket
import struct
import queue
import threading
import numpy as np

# GUI Imports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Check faster_whisper availability
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# Protocol Constants
PROTOCOL_HEARTBEAT   = 0x00
PROTOCOL_SYN         = 0x01
PROTOCOL_SYN_ACK     = 0x06
PROTOCOL_SYN_DENIED  = 0x07
PROTOCOL_SYN_PENDING  = 0x08
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END  = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Server Configuration Defaults
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_PORT = 8088
WHISPER_MODEL_NAME = "tiny"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4
BEAM_SIZE = 1

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            DEFAULT_PORT = cfg.get("tcp_port", DEFAULT_PORT)
            WHISPER_MODEL_NAME = cfg.get("whisper_model", WHISPER_MODEL_NAME)
            COMPUTE_TYPE = cfg.get("compute_type", COMPUTE_TYPE)
            CPU_THREADS = cfg.get("cpu_threads", CPU_THREADS)
    except Exception:
        pass


def recv_exact(sock, n):
    """Helper to receive exactly n bytes from a socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class STTServerEngine:
    """Core TCP Server and STT Inference Engine."""

    def __init__(self, host="0.0.0.0", port=DEFAULT_PORT, ui_queue=None):
        self.host = host
        self.port = port
        self.ui_queue = ui_queue
        self.running = False
        self.server_socket = None
        self.whisper_engine = None
        self.connected_clients = {}  # {client_key: dict_info}
        self.lock = threading.Lock()
        self.total_requests = 0

    def log_event(self, event_type, data):
        """Thread-safe dispatch to GUI queue."""
        if self.ui_queue:
            self.ui_queue.put((event_type, data))

    def load_stt_model(self):
        """Load and warm up Faster-Whisper ASR Model."""
        self.log_event("status", "⏳ Loading Faster-Whisper STT Model...")
        if HAS_FASTER_WHISPER:
            try:
                self.whisper_engine = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device="cpu",
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=CPU_THREADS
                )
                self.log_event("status", f"⚡ Model '{WHISPER_MODEL_NAME}' ({COMPUTE_TYPE}) Loaded! Warming engine...")
                
                # Warm-up inference
                t_start = time.perf_counter()
                dummy_audio = np.zeros(16000, dtype=np.float32)
                _ = self.whisper_engine.transcribe(
                    dummy_audio,
                    beam_size=1,
                    best_of=1,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=False
                )
                t_warm = (time.perf_counter() - t_start) * 1000
                self.log_event("status", f"🟢 STT Engine Ready! Pre-warmed in {t_warm:.1f} ms.")
            except Exception as e:
                self.whisper_engine = None
                self.log_event("status", f"⚠️ STT Model Load Error: {e}. Running in simulation mode.")
        else:
            self.whisper_engine = None
            self.log_event("status", "⚠️ faster-whisper not installed. Simulation mode active.")

    def start(self):
        """Start socket server thread."""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(15)
            self.log_event("status", f"🟢 Server Online! Listening on {self.host}:{self.port}")
        except Exception as e:
            self.running = False
            self.log_event("status", f"❌ Failed to bind port {self.port}: {e}")
            return False

        threading.Thread(target=self._accept_loop, daemon=True).start()
        return True

    def stop(self):
        """Stop server and close all client connections."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        self.log_event("status", "🔴 Server Shutting Down...")

    def _accept_loop(self):
        """Accept incoming client socket connections."""
        while self.running:
            try:
                client_sock, client_addr = self.server_socket.accept()
                client_key = f"{client_addr[0]}:{client_addr[1]}"
                
                with self.lock:
                    client_info = {
                        "ip": client_addr[0],
                        "port": client_addr[1],
                        "key": client_key,
                        "connected_at": time.strftime("%H:%M:%S"),
                        "connect_timestamp": time.time(),
                        "status": "ONLINE",
                        "stream_count": 0,
                        "last_transfer_ms": 0,
                        "last_stt_ms": 0,
                        "last_total_ms": 0
                    }
                    self.connected_clients[client_key] = client_info

                self.log_event("client_connect", client_info)
                threading.Thread(target=self._handle_client, args=(client_sock, client_addr, client_info), daemon=True).start()
            except Exception:
                break

    def _handle_client(self, client_sock, client_addr, client_info):
        """Process incoming client commands, audio streaming, latency timing, and STT."""
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_sock.settimeout(60.0)
        client_key = client_info["key"]

        try:
            pcm_chunks = []
            t_stream_start = None
            t_stream_end = None

            while self.running:
                try:
                    opcode_byte = recv_exact(client_sock, 1)
                except (socket.timeout, Exception):
                    break

                if not opcode_byte:
                    break

                opcode = opcode_byte[0]

                if opcode == PROTOCOL_HEARTBEAT:
                    continue

                if opcode == PROTOCOL_SYN:
                    client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
                    client_info["status"] = "AUTHENTICATED"
                    self.log_event("client_update", client_info)
                    continue

                elif opcode == PROTOCOL_AUDIO_CHUNK:
                    if t_stream_start is None:
                        t_stream_start = time.perf_counter()
                        client_info["status"] = "RECEIVING STREAM"
                        self.log_event("client_update", client_info)

                    len_bytes = recv_exact(client_sock, 2)
                    if not len_bytes or len(len_bytes) < 2:
                        break
                    payload_len = struct.unpack("<H", len_bytes)[0]
                    chunk_data = recv_exact(client_sock, payload_len)
                    if chunk_data is None:
                        break
                    pcm_chunks.append(chunk_data)

                elif opcode == PROTOCOL_STREAM_END:
                    t_stream_end = time.perf_counter()
                    # Consume length header (2 bytes)
                    _ = recv_exact(client_sock, 2)

                    # 1. Immediate Transit ACK to client
                    client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))

                    # Calculate Data Transfer Latency (Client -> Server arrival)
                    if t_stream_start and t_stream_end:
                        data_transfer_ms = int((t_stream_end - t_stream_start) * 1000)
                    else:
                        data_transfer_ms = 5  # Direct ACK timing approximation

                    # Process STT
                    raw_bytes = b"".join(pcm_chunks)
                    audio_dur_ms = int(len(raw_bytes) / 32)  # 16kHz 16-bit mono = 32 bytes/ms

                    t_stt_start = time.perf_counter()
                    transcribed_text = ""
                    detected_lang = "en"

                    if len(raw_bytes) > 0:
                        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        # Noise floor check & smart peak normalization
                        max_peak = np.max(np.abs(audio_np))
                        if max_peak > 0.008:
                            audio_np = audio_np / max_peak

                        if max_peak < 0.003:
                            transcribed_text = ""
                            detected_lang = "en"
                        elif self.whisper_engine:
                            segments, info = self.whisper_engine.transcribe(
                                audio_np,
                                beam_size=BEAM_SIZE,
                                best_of=BEAM_SIZE,
                                condition_on_previous_text=False,
                                temperature=0.0,
                                vad_filter=False,
                                initial_prompt="English and Hindi (Latin/Hinglish) smart assistant commands: turn on light, fan, switch, pankha, batti, chalao, band karo, namaste."
                            )
                            detected_lang = getattr(info, 'language', 'en')
                            transcribed_text = " ".join([seg.text for seg in segments]).strip()
                        else:
                            time.sleep(0.040)
                            transcribed_text = "Simulated Audio STT Output (Fast Whisper missing)"

                    t_stt_end = time.perf_counter()
                    stt_compute_ms = int((t_stt_end - t_stt_start) * 1000)
                    total_latency_ms = data_transfer_ms + stt_compute_ms

                    # Send Telemetry payload back to client socket (<IIIIH + text)
                    text_bytes = transcribed_text.encode('utf-8')
                    text_len = len(text_bytes)
                    try:
                        telemetry_payload = struct.pack(
                            f"<IIIIH{text_len}s",
                            audio_dur_ms,
                            0,                 # edge_ms
                            data_transfer_ms,  # net_transit_ms
                            stt_compute_ms,    # server_asr_ms
                            text_len,
                            text_bytes
                        )
                        client_sock.sendall(telemetry_payload)
                    except Exception:
                        pass

                    # Update statistics
                    with self.lock:
                        self.total_requests += 1
                        client_info["stream_count"] += 1
                        client_info["last_transfer_ms"] = data_transfer_ms
                        client_info["last_stt_ms"] = stt_compute_ms
                        client_info["last_total_ms"] = total_latency_ms
                        client_info["status"] = "IDLE"

                    transcribe_result = {
                        "client_ip": client_info["ip"],
                        "timestamp": time.strftime("%H:%M:%S"),
                        "lang": detected_lang,
                        "audio_dur_ms": audio_dur_ms,
                        "data_transfer_ms": data_transfer_ms,
                        "stt_compute_ms": stt_compute_ms,
                        "total_latency_ms": total_latency_ms,
                        "text": transcribed_text,
                        "total_requests": self.total_requests
                    }

                    self.log_event("stt_result", transcribe_result)
                    self.log_event("client_update", client_info)

                    # Reset stream buffer
                    pcm_chunks = []
                    t_stream_start = None
                    t_stream_end = None

                else:
                    # Fallback raw data ingest
                    raw_data = client_sock.recv(1024)
                    if not raw_data:
                        break
                    pcm_chunks.append(raw_data)

        except Exception as e:
            pass
        finally:
            with self.lock:
                if client_key in self.connected_clients:
                    self.connected_clients[client_key]["status"] = "DISCONNECTED"
                    self.log_event("client_disconnect", self.connected_clients[client_key])
                    del self.connected_clients[client_key]
            try:
                client_sock.close()
            except Exception:
                pass


class STTServerGUI:
    """Tkinter Dark Mode GUI Dashboard."""

    def __init__(self, root):
        self.root = root
        self.root.title("Infinix Low-Latency STT Server & Device Dashboard")
        self.root.geometry("1100x720")
        self.root.minsize(950, 600)
        self.root.configure(bg="#121824")  # Deep Dark Navy Blue

        self.ui_queue = queue.Queue()
        self.engine = STTServerEngine(port=DEFAULT_PORT, ui_queue=self.ui_queue)

        self._configure_styles()
        self._build_ui()

        # Start queue poller
        self.root.after(100, self._process_queue)

        # Load STT model in background thread
        threading.Thread(target=self.engine.load_stt_model, daemon=True).start()

        # Auto-start TCP STT server on port 8088
        self.root.after(200, self.toggle_server)

    def _configure_styles(self):
        """Custom dark theme styles for ttk components."""
        self.style = ttk.Style()
        self.style.theme_use("default")

        self.style.configure(".", background="#121824", foreground="#e0e6ed", font=("Segoe UI", 10))
        
        # Frames & Cards
        self.style.configure("TFrame", background="#121824")
        self.style.configure("Card.TFrame", background="#1e2638", relief="flat")
        
        # Labels
        self.style.configure("TLabel", background="#121824", foreground="#e0e6ed")
        self.style.configure("CardTitle.TLabel", background="#1e2638", foreground="#8a99ad", font=("Segoe UI", 9, "bold"))
        self.style.configure("CardVal.TLabel", background="#1e2638", foreground="#00e5ff", font=("Segoe UI", 18, "bold"))
        self.style.configure("Header.TLabel", background="#121824", foreground="#ffffff", font=("Segoe UI", 16, "bold"))

        # Buttons
        self.style.configure("Accent.TButton", background="#00e5ff", foreground="#000000", font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("Accent.TButton", background=[("active", "#00b2cc"), ("disabled", "#334155")])

        self.style.configure("Stop.TButton", background="#ff5252", foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("Stop.TButton", background=[("active", "#d32f2f")])

        # Treeview (Client & STT Tables)
        self.style.configure("Treeview", background="#1a2233", foreground="#e0e6ed", fieldbackground="#1a2233", rowheight=28, font=("Segoe UI", 9))
        self.style.configure("Treeview.Heading", background="#28354e", foreground="#00e5ff", font=("Segoe UI", 9, "bold"))
        self.style.map("Treeview", background=[("selected", "#00e5ff")], foreground=[("selected", "#000000")])

    def _build_ui(self):
        """Construct GUI layout."""
        # Top Header Bar
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=15, pady=10)

        lbl_title = ttk.Label(header_frame, text="⚡ Low-Latency STT Server Dashboard", style="Header.TLabel")
        lbl_title.pack(side="left")

        self.lbl_server_status = tk.Label(
            header_frame, text="🔴 OFFLINE", bg="#ff5252", fg="#ffffff",
            font=("Segoe UI", 10, "bold"), padx=12, pady=4, relief="flat"
        )
        self.lbl_server_status.pack(side="right")

        self.btn_toggle_server = ttk.Button(header_frame, text="▶ Start Server", style="Accent.TButton", command=self.toggle_server)
        self.btn_toggle_server.pack(side="right", padx=10)

        self.btn_test_sim = ttk.Button(header_frame, text="🧪 Test Latency", command=self.run_test_simulation)
        self.btn_test_sim.pack(side="right", padx=5)

        # Performance Metrics Summary Cards Bar
        cards_frame = ttk.Frame(self.root)
        cards_frame.pack(fill="x", padx=15, pady=5)

        # Card 1: Active Devices
        c1 = ttk.Frame(cards_frame, style="Card.TFrame", padding=10)
        c1.pack(side="left", fill="both", expand=True, padx=4)
        ttk.Label(c1, text="ACTIVE DEVICES", style="CardTitle.TLabel").pack(anchor="w")
        self.lbl_card_devices = ttk.Label(c1, text="0 Connected", style="CardVal.TLabel")
        self.lbl_card_devices.pack(anchor="w", pady=2)

        # Card 2: Total Requests
        c2 = ttk.Frame(cards_frame, style="Card.TFrame", padding=10)
        c2.pack(side="left", fill="both", expand=True, padx=4)
        ttk.Label(c2, text="TOTAL REQUESTS", style="CardTitle.TLabel").pack(anchor="w")
        self.lbl_card_requests = ttk.Label(c2, text="0", style="CardVal.TLabel")
        self.lbl_card_requests.pack(anchor="w", pady=2)

        # Card 3: Data Transfer Latency
        c3 = ttk.Frame(cards_frame, style="Card.TFrame", padding=10)
        c3.pack(side="left", fill="both", expand=True, padx=4)
        ttk.Label(c3, text="DATA TRANSFER (CLIENT->SERVER)", style="CardTitle.TLabel").pack(anchor="w")
        self.lbl_card_transfer = ttk.Label(c3, text="0 ms", style="CardVal.TLabel")
        self.lbl_card_transfer.configure(foreground="#ffb74d")
        self.lbl_card_transfer.pack(anchor="w", pady=2)

        # Card 4: STT Latency
        c4 = ttk.Frame(cards_frame, style="Card.TFrame", padding=10)
        c4.pack(side="left", fill="both", expand=True, padx=4)
        ttk.Label(c4, text="SERVER STT COMPUTE", style="CardTitle.TLabel").pack(anchor="w")
        self.lbl_card_stt = ttk.Label(c4, text="0 ms", style="CardVal.TLabel")
        self.lbl_card_stt.configure(foreground="#00e676")
        self.lbl_card_stt.pack(anchor="w", pady=2)

        # Main Paned Notebook / View Areas
        main_paned = tk.PanedWindow(self.root, orient="vertical", bg="#121824", bd=0, sashwidth=4)
        main_paned.pack(fill="both", expand=True, padx=15, pady=10)

        # Upper Area: Connected Devices List
        dev_frame = ttk.LabelFrame(main_paned, text=" 📱 Connected Devices ", padding=8)
        main_paned.add(dev_frame, height=200)

        cols_dev = ("ip_port", "status", "connected_at", "streams", "last_transfer", "last_stt", "last_total")
        self.tree_dev = ttk.Treeview(dev_frame, columns=cols_dev, show="headings", height=5)
        self.tree_dev.heading("ip_port", text="Client IP:Port")
        self.tree_dev.heading("status", text="Status")
        self.tree_dev.heading("connected_at", text="Connected At")
        self.tree_dev.heading("streams", text="Streams Sent")
        self.tree_dev.heading("last_transfer", text="Data Transfer (ms)")
        self.tree_dev.heading("last_stt", text="STT Compute (ms)")
        self.tree_dev.heading("last_total", text="Total Latency (ms)")

        self.tree_dev.column("ip_port", width=160, anchor="center")
        self.tree_dev.column("status", width=120, anchor="center")
        self.tree_dev.column("connected_at", width=110, anchor="center")
        self.tree_dev.column("streams", width=100, anchor="center")
        self.tree_dev.column("last_transfer", width=140, anchor="center")
        self.tree_dev.column("last_stt", width=140, anchor="center")
        self.tree_dev.column("last_total", width=140, anchor="center")
        self.tree_dev.pack(fill="both", expand=True)

        # Lower Area: STT Log & Live Output Table
        log_frame = ttk.LabelFrame(main_paned, text=" 🎙️ Speech-To-Text (STT) & Latency Feed ", padding=8)
        main_paned.add(log_frame, height=360)

        cols_log = ("time", "client", "lang", "audio_dur", "transfer_ms", "stt_ms", "total_ms", "text")
        self.tree_log = ttk.Treeview(log_frame, columns=cols_log, show="headings")
        self.tree_log.heading("time", text="Timestamp")
        self.tree_log.heading("client", text="Client IP")
        self.tree_log.heading("lang", text="Lang")
        self.tree_log.heading("audio_dur", text="Audio (ms)")
        self.tree_log.heading("transfer_ms", text="Transfer (ms)")
        self.tree_log.heading("stt_ms", text="STT Compute (ms)")
        self.tree_log.heading("total_ms", text="Total Latency (ms)")
        self.tree_log.heading("text", text="Transcribed Speech Text")

        self.tree_log.column("time", width=90, anchor="center")
        self.tree_log.column("client", width=120, anchor="center")
        self.tree_log.column("lang", width=70, anchor="center")
        self.tree_log.column("audio_dur", width=90, anchor="center")
        self.tree_log.column("transfer_ms", width=100, anchor="center")
        self.tree_log.column("stt_ms", width=110, anchor="center")
        self.tree_log.column("total_ms", width=110, anchor="center")
        self.tree_log.column("text", width=380, anchor="w")

        scrollbar_log = ttk.Scrollbar(log_frame, orient="vertical", command=self.tree_log.yview)
        self.tree_log.configure(yscrollcommand=scrollbar_log.set)
        scrollbar_log.pack(side="right", fill="y")
        self.tree_log.pack(fill="both", expand=True)

        # Bottom Status Bar
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", padx=15, pady=5)
        self.lbl_status_msg = ttk.Label(status_bar, text="Initializing GUI...", font=("Segoe UI", 9, "italic"))
        self.lbl_status_msg.pack(side="left")

    def toggle_server(self):
        """Start or stop the TCP STT server."""
        if not self.engine.running:
            success = self.engine.start()
            if success:
                self.lbl_server_status.configure(text=f"🟢 ONLINE (PORT {self.engine.port})", bg="#00e676", fg="#000000")
                self.btn_toggle_server.configure(text="⏹ Stop Server", style="Stop.TButton")
        else:
            self.engine.stop()
            self.lbl_server_status.configure(text="🔴 OFFLINE", bg="#ff5252", fg="#ffffff")
            self.btn_toggle_server.configure(text="▶ Start Server", style="Accent.TButton")

    def run_test_simulation(self):
        """Simulate an internal client sending audio to test latencies."""
        if not self.engine.running:
            messagebox.showwarning("Server Offline", "Please start the server first before running the latency test simulation!")
            return

        def _sim():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect(("127.0.0.1", self.engine.port))

                # Send SYN Handshake
                sock.sendall(bytes([PROTOCOL_SYN]))
                _ = sock.recv(1)

                # Stream 1.5 seconds of PCM audio
                sample_rate = 16000
                duration_sec = 1.5
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
                tone = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
                audio_bytes = tone.tobytes()

                chunk_size = 512
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(chunk))
                    sock.sendall(header + chunk)
                    time.sleep(0.004)

                # Send Stream End
                end_header = struct.pack("<BH", PROTOCOL_STREAM_END, 0)
                sock.sendall(end_header)

                # Read ACK & Telemetry
                _ = sock.recv(1)
                _ = sock.recv(18)
                sock.close()
            except Exception as e:
                pass

        threading.Thread(target=_sim, daemon=True).start()

    def _process_queue(self):
        """Process thread-safe UI updates from the queue."""
        while not self.ui_queue.empty():
            try:
                event_type, data = self.ui_queue.get_nowait()
                
                if event_type == "status":
                    self.lbl_status_msg.configure(text=data)

                elif event_type in ("client_connect", "client_update", "client_disconnect"):
                    self._update_client_tree(data)

                elif event_type == "stt_result":
                    self._add_stt_log(data)

            except queue.Empty:
                break

        self.root.after(100, self._process_queue)

    def _update_client_tree(self, client_info):
        """Update or insert client device entry in treeview."""
        key = client_info["key"]
        values = (
            key,
            client_info["status"],
            client_info["connected_at"],
            client_info["stream_count"],
            f"{client_info['last_transfer_ms']} ms",
            f"{client_info['last_stt_ms']} ms",
            f"{client_info['last_total_ms']} ms"
        )

        existing = None
        for item in self.tree_dev.get_children():
            if self.tree_dev.item(item, "values")[0] == key:
                existing = item
                break

        if client_info["status"] == "DISCONNECTED":
            if existing:
                self.tree_dev.delete(existing)
        else:
            if existing:
                self.tree_dev.item(existing, values=values)
            else:
                self.tree_dev.insert("", "end", values=values)

        active_count = len(self.tree_dev.get_children())
        self.lbl_card_devices.configure(text=f"{active_count} Active")

    def _add_stt_log(self, data):
        """Add transcription log item and update metrics cards."""
        lang_icon = "🇮🇳 HI" if data["lang"] == "hi" else ("🇬🇧 EN" if data["lang"] == "en" else data["lang"].upper())
        values = (
            data["timestamp"],
            data["client_ip"],
            lang_icon,
            f"{data['audio_dur_ms']} ms",
            f"{data['data_transfer_ms']} ms",
            f"{data['stt_compute_ms']} ms",
            f"{data['total_latency_ms']} ms",
            data["text"] if data["text"] else "[Silence / No speech detected]"
        )

        self.tree_log.insert("", 0, values=values)  # Prepend newest on top

        # Update Card Displays
        self.lbl_card_requests.configure(text=str(data["total_requests"]))
        self.lbl_card_transfer.configure(text=f"{data['data_transfer_ms']} ms")
        self.lbl_card_stt.configure(text=f"{data['stt_compute_ms']} ms")


def main():
    root = tk.Tk()
    app = STTServerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
