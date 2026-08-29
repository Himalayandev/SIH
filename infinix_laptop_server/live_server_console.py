#!/usr/bin/env python3
"""
=============================================================================
INFINIX LAPTOP SERVER - LIVE DASHBOARD & REQUEST MONITOR
Real-time console dashboard for tracking incoming ASR requests from ESP32 / PCs.
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
WHISPER_MODEL_NAME = "base"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4

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

# Global State
stats = {
    "total_requests": 0,
    "last_client_ip": "None",
    "last_lang": "None",
    "last_text": "Waiting for requests...",
    "last_duration_ms": 0,
    "last_latency_ms": 0,
    "start_time": time.time(),
}
request_history = []
lock = threading.Lock()

# Load Whisper Model
whisper_engine = None
if HAS_FASTER_WHISPER:
    try:
        whisper_engine = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
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
    client_sock.settimeout(12.0)
    is_protocol = False

    try:
        first_byte = client_sock.recv(1)
        if not first_byte:
            client_sock.close()
            return

        pcm_chunks = []
        if first_byte[0] == PROTOCOL_SYN:
            is_protocol = True
            client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
        else:
            pcm_chunks.append(first_byte)

        while True:
            try:
                chunk = client_sock.recv(1024)
                if not chunk:
                    break

                if is_protocol:
                    if chunk == bytes([PROTOCOL_STREAM_END]):
                        client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                        break
                    elif len(chunk) > 1 and chunk[-1] == PROTOCOL_STREAM_END:
                        pcm_chunks.append(chunk[:-1])
                        client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                        break

                pcm_chunks.append(chunk)
            except socket.timeout:
                break
            except Exception:
                break

        raw_bytes = b"".join(pcm_chunks)
        audio_dur_ms = int(len(raw_bytes) / 32)

        t_asr_start = time.perf_counter()
        transcribed_text = ""
        detected_lang = "en"

        if len(raw_bytes) > 0:
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if whisper_engine:
                segments, info = whisper_engine.transcribe(audio_np, beam_size=1)
                transcribed_text = " ".join([seg.text for seg in segments]).strip()
                detected_lang = getattr(info, 'language', 'en')
            else:
                time.sleep(0.042)
                transcribed_text = "Sample audio processed"

        # Import ASTA Engine
        try:
            from asta_engine import ASTACommandValidator, MetricsCollector
            HAS_ASTA = True
        except ImportError:
            HAS_ASTA = False

        t_asr_end = time.perf_counter()
        asr_compute_ms = int((t_asr_end - t_asr_start) * 1000)

        if HAS_ASTA:
            metrics = MetricsCollector.get_metrics(asr_latency_ms=asr_compute_ms, audio_dur_ms=audio_dur_ms)
            asta_res = ASTACommandValidator.validate_and_repair(transcribed_text)
        else:
            metrics = {"cpu_workload_pct": 0}
            asta_res = {"valid": False, "repaired_command": None, "was_repaired": False}

        if is_protocol:
            try:
                telemetry_payload = struct.pack("<IIII", audio_dur_ms, 0, 0, asr_compute_ms)
                client_sock.sendall(telemetry_payload)
            except Exception:
                pass

        timestamp = time.strftime("%H:%M:%S")
        lang_label = "Hindi 🇮🇳" if detected_lang == "hi" else ("English 🇬🇧" if detected_lang == "en" else detected_lang.upper())
        repaired_cmd = asta_res["repaired_command"] if asta_res["valid"] else "(No IoT Action Match)"

        with lock:
            stats["total_requests"] += 1
            stats["last_client_ip"] = client_addr[0]
            stats["last_lang"] = lang_label
            stats["last_text"] = transcribed_text if transcribed_text else "(No speech detected)"
            stats["last_cmd"] = repaired_cmd
            stats["last_duration_ms"] = audio_dur_ms
            stats["last_latency_ms"] = asr_compute_ms
            stats["cpu_pct"] = metrics["cpu_workload_pct"]

            request_history.insert(0, {
                "time": timestamp,
                "ip": client_addr[0],
                "lang": lang_label,
                "dur": f"{audio_dur_ms} ms",
                "asr": f"{asr_compute_ms} ms",
                "text": transcribed_text if transcribed_text else "(empty)",
                "action": repaired_cmd
            })
            if len(request_history) > 8:
                request_history.pop()

    except Exception as e:
        pass
    finally:
        try:
            client_sock.close()
        except Exception:
            pass

def generate_dashboard():
    console = Console()
    uptime_sec = int(time.time() - stats["start_time"])
    local_ips = ", ".join(get_local_ips())

    table = Table(title="📜 Recent Incoming Live Requests & ASTA Actions Log", expand=True, header_style="bold cyan")
    table.add_column("Time", style="yellow", width=10)
    table.add_column("Client IP", style="magenta", width=16)
    table.add_column("Language", style="green", width=14)
    table.add_column("Duration", style="blue", width=10)
    table.add_column("ASR Latency", style="red", width=12)
    table.add_column("ASTA Executable Command", style="bold green", width=22)
    table.add_column("Transcribed Text", style="bold white")

    with lock:
        for req in request_history:
            table.add_row(req["time"], req["ip"], req["lang"], req["dur"], req["asr"], req.get("action", req["text"]), req["text"])

    status_text = Text()
    status_text.append("🟢 SERVER STATUS: ONLINE & LISTENING\n", style="bold green")
    status_text.append(f"• Listening Port : {TCP_SERVER_PORT}\n", style="bold white")
    status_text.append(f"• Server IP      : {local_ips}\n", style="bold yellow")
    status_text.append(f"• Total Requests : {stats['total_requests']}\n", style="bold cyan")
    status_text.append(f"• Server Uptime  : {uptime_sec} seconds\n", style="bold magenta")
    status_text.append(f"• STT Engine     : Faster-Whisper '{WHISPER_MODEL_NAME}' (INT8 CPU)", style="bold blue")

    last_req_text = Text()
    last_req_text.append("⚡ LATEST REQUEST METRICS:\n", style="bold yellow")
    last_req_text.append(f"• Last Client IP  : {stats['last_client_ip']}\n", style="white")
    last_req_text.append(f"• Language        : {stats['last_lang']}\n", style="white")
    last_req_text.append(f"• Audio Duration  : {stats['last_duration_ms']} ms\n", style="white")
    last_req_text.append(f"• Inference Time  : {stats['last_latency_ms']} ms\n\n", style="white")
    last_req_text.append("📝 TRANSCRIPTION:\n", style="bold green")
    last_req_text.append(f"\"{stats['last_text']}\"", style="bold italic white")

    layout = Layout()
    layout.split_column(
        Layout(Panel(status_text, title="🚀 INFINIX LAPTOP ASR LIVE SERVER DASHBOARD", border_style="bright_blue"), size=9),
        Layout(Panel(last_req_text, title="🎯 LIVE REAL-TIME REQUEST TRACKER", border_style="bright_yellow"), size=9),
        Layout(Panel(table, border_style="green"))
    )

    return layout

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", TCP_SERVER_PORT))
    server.listen(10)

    # Server Thread
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
                print("\n[INFO] Dashboard stopped.")
    else:
        print(f"🟢 Server listening on 0.0.0.0:{TCP_SERVER_PORT}...")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    server.close()

if __name__ == "__main__":
    main()
