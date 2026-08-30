#!/usr/bin/env python3
"""
=============================================================================
INFINIX ASR LAPTOP SERVER - MINIMALIST SECURITY/DEVELOPER TUI DASHBOARD
Subdued, high-end monochromatic aesthetic with icy cyan borders, rounded
geometry, strict alignment, and functional alerts.
=============================================================================
"""

import os
import sys
import time
import json
import socket
import struct
import threading
import numpy as np

# Ensure UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
TCP_SERVER_PORT = 8088
WHISPER_MODEL_NAME = "tiny"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4
BEAM_SIZE = 1

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            TCP_SERVER_PORT = cfg.get("tcp_port", TCP_SERVER_PORT)
            WHISPER_MODEL_NAME = cfg.get("whisper_model", WHISPER_MODEL_NAME)
            COMPUTE_TYPE = cfg.get("compute_type", COMPUTE_TYPE)
            CPU_THREADS = cfg.get("cpu_threads", CPU_THREADS)
    except Exception:
        pass

PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Styling Constants (Minimalist Monochromatic Palette)
BORDER_COLOR = "steel_blue"       # Muted icy cyan / steel blue border accent
TEXT_PRIMARY = "#cccccc"         # Soft ash gray primary text
TEXT_SUBDUED = "#888888"         # Muted label text
ALERT_RUST = "#cc5555"           # Muted rust / dark red for high latency / high CPU load
STATUS_GREEN = "#55aa55"         # Dim, soft green for ONLINE status dot ONLY

# Global State
stats = {
    "total_requests": 0,
    "last_client_ip": "None",
    "last_lang": "None",
    "last_text": "Waiting for incoming streams...",
    "last_cmd": "None",
    "last_duration_ms": 0,
    "last_latency_ms": 0,
    "cpu_pct": 0.0,
    "start_time": time.time(),
}
request_history = []
lock = threading.Lock()

# Protocol Opcodes
PROTOCOL_HEARTBEAT = 0x00
PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Load Whisper Model
whisper_engine = None
if HAS_FASTER_WHISPER:
    try:
        whisper_engine = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
        # Dummy Warm-Up Inference (Eliminates Cold Start Latency Spike)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = whisper_engine.transcribe(
            dummy_audio,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=False
        )
    except Exception:
        whisper_engine = None

def get_local_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips if ips else ["127.0.0.1"]

def handle_client(client_sock, client_addr):
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_sock.settimeout(60.0) # Persistent pre-warmed idle connection timeout
    is_protocol = False

    try:
        pcm_chunks = []

        while True:
            try:
                opcode_byte = client_sock.recv(1)
            except socket.timeout:
                break
            except Exception:
                break

            if not opcode_byte:
                break

            opcode = opcode_byte[0]

            if opcode == PROTOCOL_HEARTBEAT:
                # Idle TCP Keepalive Ping
                continue

            if opcode == PROTOCOL_SYN:
                is_protocol = True
                client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
                continue

            elif opcode == PROTOCOL_AUDIO_CHUNK:
                # Length-Prefixed TLV Frame: Read 2-byte N (payload length)
                len_bytes = client_sock.recv(2)
                if len(len_bytes) < 2:
                    break
                payload_len = struct.unpack("<H", len_bytes)[0]

                chunk_data = b""
                while len(chunk_data) < payload_len:
                    more = client_sock.recv(payload_len - len(chunk_data))
                    if not more:
                        break
                    chunk_data += more

                pcm_chunks.append(chunk_data)

            elif opcode == PROTOCOL_STREAM_END:
                len_bytes = client_sock.recv(2) # Consume 2-byte length if present
                client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))

                raw_bytes = b"".join(pcm_chunks)
                audio_dur_ms = int(len(raw_bytes) / 32)

                t_asr_start = time.perf_counter()
                transcribed_text = ""
                detected_lang = "en"

                if len(raw_bytes) > 0:
                    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    max_peak = np.max(np.abs(audio_np))
                    if max_peak > 0.008:
                        audio_np = audio_np / max_peak

                    if max_peak < 0.003:
                        transcribed_text = ""
                        detected_lang = "en"
                    elif whisper_engine:
                        segments, info = whisper_engine.transcribe(
                            audio_np,
                            beam_size=BEAM_SIZE,
                            best_of=BEAM_SIZE,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=400),
                            initial_prompt="English and Hindi (Latin/Hinglish) smart assistant voice commands: turn on light, fan, switch, pankha, batti, chalao, band karo, namaste, kaise ho."
                        )
                        detected_lang = getattr(info, 'language', 'en')

                        if detected_lang not in ["en", "hi"]:
                            segments, info = whisper_engine.transcribe(
                                audio_np,
                                beam_size=1,
                                language="en",
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=400),
                                initial_prompt="English and Hindi (Latin/Hinglish) smart assistant voice commands: turn on light, fan, switch, pankha, batti."
                            )
                            detected_lang = "en"

                        transcribed_text = " ".join([seg.text for seg in segments]).strip()
                    else:
                        time.sleep(0.042)
                        transcribed_text = "Sample audio stream"

                t_asr_end = time.perf_counter()
                asr_compute_ms = int((t_asr_end - t_asr_start) * 1000)

                # Send 18-Byte Telemetry Header + UTF-8 Transcribed Text Payload back to ESP32 OLED
                text_bytes = transcribed_text.encode('utf-8')
                text_len = len(text_bytes)
                if is_protocol:
                    try:
                        telemetry_payload = struct.pack(f"<IIIIH{text_len}s", audio_dur_ms, 0, 0, asr_compute_ms, text_len, text_bytes)
                        client_sock.sendall(telemetry_payload)
                    except Exception:
                        pass

                timestamp = time.strftime("%H:%M:%S")
                lang_label = "Hindi (hi)" if detected_lang == "hi" else ("English (en)" if detected_lang == "en" else detected_lang.upper())

                with lock:
                    stats["total_requests"] += 1
                    stats["last_client_ip"] = client_addr[0]
                    stats["last_lang"] = lang_label
                    stats["last_text"] = transcribed_text if transcribed_text else "(empty)"
                    stats["last_duration_ms"] = audio_dur_ms
                    stats["last_latency_ms"] = asr_compute_ms

                    request_history.insert(0, {
                        "time": timestamp,
                        "ip": client_addr[0],
                        "lang": lang_label,
                        "dur_ms": audio_dur_ms,
                        "asr_ms": asr_compute_ms,
                        "text": transcribed_text if transcribed_text else "(empty)"
                    })
                    if len(request_history) > 10:
                        request_history.pop()

                pcm_chunks = []

            else:
                raw_data = client_sock.recv(1024)
                if not raw_data:
                    break
                pcm_chunks.append(raw_data)

        timestamp = time.strftime("%H:%M:%S")
        lang_label = "Hindi (hi)" if detected_lang == "hi" else ("English (en)" if detected_lang == "en" else detected_lang.upper())

        with lock:
            stats["total_requests"] += 1
            stats["last_client_ip"] = client_addr[0]
            stats["last_lang"] = lang_label
            stats["last_text"] = transcribed_text if transcribed_text else "(empty)"
            stats["last_duration_ms"] = audio_dur_ms
            stats["last_latency_ms"] = asr_compute_ms

            request_history.insert(0, {
                "time": timestamp,
                "ip": client_addr[0],
                "lang": lang_label,
                "dur_ms": audio_dur_ms,
                "asr_ms": asr_compute_ms,
                "text": transcribed_text if transcribed_text else "(empty)"
            })
            if len(request_history) > 10:
                request_history.pop()

    except Exception:
        pass
    finally:
        try:
            client_sock.close()
        except Exception:
            pass

def generate_dashboard():
    uptime_sec = int(time.time() - stats["start_time"])
    local_ips = ", ".join(get_local_ips())

    # 1. System Status Panel
    status_text = Text()
    status_text.append("● ", style=STATUS_GREEN)
    status_text.append("ONLINE & LISTENING\n", style=f"bold {TEXT_PRIMARY}")
    
    status_text.append("Port           : ", style=TEXT_SUBDUED)
    status_text.append(f"{TCP_SERVER_PORT}\n", style=TEXT_PRIMARY)
    
    status_text.append("Server IP      : ", style=TEXT_SUBDUED)
    status_text.append(f"{local_ips}\n", style=TEXT_PRIMARY)
    
    status_text.append("Total Requests : ", style=TEXT_SUBDUED)
    status_text.append(f"{stats['total_requests']}\n", style=TEXT_PRIMARY)
    
    status_text.append("ASR Engine     : ", style=TEXT_SUBDUED)
    status_text.append(f"Faster-Whisper '{WHISPER_MODEL_NAME}' (INT8 CPU)", style=TEXT_PRIMARY)

    panel_status = Panel(
        status_text,
        title=" SYSTEM STATUS ",
        title_align="left",
        border_style=BORDER_COLOR,
        box=box.ROUNDED,
        padding=(1, 2)
    )

    # 2. Latest Request Metrics Panel
    metrics_text = Text()
    metrics_text.append("Client IP      : ", style=TEXT_SUBDUED)
    metrics_text.append(f"{stats['last_client_ip']}\n", style=TEXT_PRIMARY)
    
    metrics_text.append("Language       : ", style=TEXT_SUBDUED)
    metrics_text.append(f"{stats['last_lang']}\n", style=TEXT_PRIMARY)
    
    metrics_text.append("Audio Duration : ", style=TEXT_SUBDUED)
    metrics_text.append(f"{stats['last_duration_ms']} ms\n", style=TEXT_PRIMARY)
    
    metrics_text.append("ASR Compute    : ", style=TEXT_SUBDUED)
    latency_style = ALERT_RUST if stats['last_latency_ms'] > 1000 else TEXT_PRIMARY
    metrics_text.append(f"{stats['last_latency_ms']} ms\n\n", style=latency_style)
    
    metrics_text.append("Speech Output  : ", style=TEXT_SUBDUED)
    metrics_text.append(f"\"{stats['last_text']}\"", style=TEXT_PRIMARY)

    panel_metrics = Panel(
        metrics_text,
        title=" LATEST REQUEST METRICS ",
        title_align="left",
        border_style=BORDER_COLOR,
        box=box.ROUNDED,
        padding=(1, 2)
    )

    # 3. Request Log Table (Partitioned rows with horizontal dividers)
    table = Table(
        expand=True,
        box=box.HORIZONTALS,
        show_lines=True,
        header_style=f"bold {TEXT_PRIMARY}",
        border_style="grey35",
        padding=(0, 1)
    )
    
    table.add_column("Time", justify="left", style=TEXT_PRIMARY, width=10)
    table.add_column("Client IP", justify="left", style=TEXT_PRIMARY, width=16)
    table.add_column("Language", justify="left", style=TEXT_PRIMARY, width=14)
    table.add_column("Duration", justify="right", style=TEXT_PRIMARY, width=10)
    table.add_column("ASR Latency", justify="right", width=12)
    table.add_column("Transcribed Text", justify="left", style=TEXT_PRIMARY, no_wrap=True, overflow="ellipsis")

    with lock:
        for req in request_history:
            lat_style = ALERT_RUST if req["asr_ms"] > 1000 else TEXT_PRIMARY
            lat_text = Text(f"{req['asr_ms']} ms", style=lat_style)
            
            table.add_row(
                req["time"],
                req["ip"],
                req["lang"],
                f"{req['dur_ms']} ms",
                lat_text,
                req["text"]
            )

    panel_table = Panel(
        table,
        title=" INCOMING REQUESTS LOG ",
        title_align="left",
        border_style=BORDER_COLOR,
        box=box.ROUNDED,
        padding=(1, 2)
    )

    layout = Layout()
    layout.split_column(
        Layout(name="top_panels", size=10),
        Layout(panel_table, name="bottom_table")
    )
    
    layout["top_panels"].split_row(
        Layout(panel_status, ratio=1),
        Layout(panel_metrics, ratio=1)
    )

    return layout

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", TCP_SERVER_PORT))
    server.listen(10)

    def listen_loop():
        while True:
            try:
                client_sock, client_addr = server.accept()
                t = threading.Thread(target=handle_client, args=(client_sock, client_addr), daemon=True)
                t.start()
            except Exception:
                break

    t_listen = threading.Thread(target=listen_loop, daemon=True)
    t_listen.start()

    if HAS_RICH:
        console = Console()
        with Live(generate_dashboard(), console=console, refresh_per_second=4, screen=True) as live:
            try:
                while True:
                    time.sleep(0.25)
                    live.update(generate_dashboard())
            except KeyboardInterrupt:
                pass
    else:
        print(f"Server listening on 0.0.0.0:{TCP_SERVER_PORT}...")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    server.close()

if __name__ == "__main__":
    main()
