#!/usr/bin/env python3
"""
=============================================================================
INFINIX LAPTOP SERVER - REAL-TIME ASR & LIVE LOGGING ENGINE
Features: Instant stdout flushing, persistent rolling file logs (server_live.log),
Faster-Whisper (INT8 CPU) ASR, and zero-delay binary TCP sockets.
=============================================================================
"""

import os
import sys
import json
import socket
import struct
import time
import threading
import numpy as np

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "server_live.log")

def log_msg(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    formatted_msg = f"{timestamp} {msg}"
    print(formatted_msg, flush=True)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
            f.flush()
    except Exception:
        pass

# Try importing faster_whisper
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# Load Config if present
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
TCP_SERVER_PORT = 8088
WHISPER_MODEL_NAME = "small"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4
BEAM_SIZE = 5

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            TCP_SERVER_PORT = cfg.get("tcp_port", TCP_SERVER_PORT)
            WHISPER_MODEL_NAME = cfg.get("whisper_model", WHISPER_MODEL_NAME)
            COMPUTE_TYPE = cfg.get("compute_type", COMPUTE_TYPE)
            CPU_THREADS = cfg.get("cpu_threads", CPU_THREADS)
            BEAM_SIZE = cfg.get("beam_size", BEAM_SIZE)
    except Exception as e:
        log_msg(f"⚠️ Failed to parse config.json: {e}")

# Protocol Constants
PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

log_msg("=" * 60)
log_msg("🚀 [1/2] Loading Fast Offline Whisper ASR Engine...")
if HAS_FASTER_WHISPER:
    whisper_engine = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    log_msg(f"✅ Model loaded: Faster-Whisper '{WHISPER_MODEL_NAME}' ({COMPUTE_TYPE.upper()} Quantized, {CPU_THREADS} threads)")
else:
    whisper_engine = None
    log_msg("⚠️ faster-whisper not installed. Running in high-speed simulation mode.")
log_msg("=" * 60)

def handle_esp32_connection(client_sock, client_addr):
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_sock.settimeout(12.0)
    is_protocol_client = False

    log_msg(f"⚡ [INCOMING CONNECT] Connection initiated from IP: {client_addr[0]}:{client_addr[1]}")

    try:
        first_byte = client_sock.recv(1)
        if not first_byte:
            log_msg(f"⚠️ [DISCONNECT] {client_addr[0]} closed socket without data.")
            client_sock.close()
            return

        pcm_chunks = []

        if first_byte[0] == PROTOCOL_SYN:
            is_protocol_client = True
            client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
            log_msg(f"🤝 [HANDSHAKE] Binary SYN 0x01 verified. Sent SYN-ACK 0x06 to {client_addr[0]}")
        else:
            log_msg(f"🎙️ [RAW STREAM] Direct audio stream started from {client_addr[0]}")
            pcm_chunks.append(first_byte)

        # Ingest remaining audio data
        while True:
            try:
                chunk = client_sock.recv(1024)
                if not chunk:
                    break

                if is_protocol_client:
                    if chunk == bytes([PROTOCOL_STREAM_END]):
                        client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                        log_msg(f"🚀 [TRANSIT ACK] Sent 0x7F instant hardware ACK to {client_addr[0]}")
                        break
                    elif len(chunk) > 1 and chunk[-1] == PROTOCOL_STREAM_END:
                        pcm_chunks.append(chunk[:-1])
                        client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                        log_msg(f"🚀 [TRANSIT ACK] Sent 0x7F instant hardware ACK to {client_addr[0]}")
                        break

                pcm_chunks.append(chunk)
            except socket.timeout:
                log_msg(f"⏱️ [STREAM TIMEOUT] Stopped reading from {client_addr[0]} (Timeout)")
                break
            except Exception:
                break

        raw_bytes = b"".join(pcm_chunks)
        audio_dur_ms = int(len(raw_bytes) / 32)
        log_msg(f"📥 [AUDIO RECEIVED] Total bytes: {len(raw_bytes)} ({audio_dur_ms} ms) from {client_addr[0]}")

        # Transcribe with Faster-Whisper (Enhanced Audio Processing & Beam Search)
        t_asr_start = time.perf_counter()
        transcribed_text = ""
        detected_lang = "en"

        if len(raw_bytes) > 0:
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 1. Peak Normalization (Boost low-volume mic inputs to prevent dropped words)
            max_peak = np.max(np.abs(audio_np))
            if max_peak > 0.001:
                audio_np = audio_np / max_peak

            if whisper_engine:
                # 2. Advanced High-Accuracy Beam Decoding & VAD Noise Filter
                segments, info = whisper_engine.transcribe(
                    audio_np,
                    beam_size=BEAM_SIZE,
                    best_of=BEAM_SIZE,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400),
                    initial_prompt="Hindi and English smart assistant commands: turn on light, fan, switch, activate, namaste, kaise ho."
                )
                transcribed_text = " ".join([seg.text for seg in segments]).strip()
                detected_lang = getattr(info, 'language', 'en')
            else:
                time.sleep(0.042)
                transcribed_text = "Sample audio processed"

        # Import ASTA Engine
        try:
            from asta_engine import ASTACommandValidator, MetricsCollector, ASTARouter
            HAS_ASTA = True
        except ImportError:
            HAS_ASTA = False

        t_asr_end = time.perf_counter()
        asr_compute_ms = int((t_asr_end - t_asr_start) * 1000)

        # ASTA System Metrics & Command Repair Processing
        if HAS_ASTA:
            metrics = MetricsCollector.get_metrics(asr_latency_ms=asr_compute_ms, audio_dur_ms=audio_dur_ms)
            route_mode = ASTARouter.route(metrics)
            asta_res = ASTACommandValidator.validate_and_repair(transcribed_text)
        else:
            metrics = {"cpu_workload_pct": 0, "status": "NORMAL"}
            route_mode = "LOCAL_EDGE"
            asta_res = {"valid": False, "repaired_command": None, "was_repaired": False}

        if is_protocol_client:
            try:
                telemetry_payload = struct.pack("<IIII", audio_dur_ms, 0, 0, asr_compute_ms)
                client_sock.sendall(telemetry_payload)
            except Exception:
                pass

        lang_label = "Hindi 🇮🇳" if detected_lang == "hi" else ("English 🇬🇧" if detected_lang == "en" else detected_lang.upper())

        log_msg(f"✅ [STT COMPLETED] Client: {client_addr[0]} | Lang: {lang_label} | Audio: {audio_dur_ms}ms | Latency: {asr_compute_ms}ms | CPU: {metrics['cpu_workload_pct']}% | Text: \"{transcribed_text}\"")
        if asta_res["valid"]:
            repair_str = f" (Auto-Repaired from history)" if asta_res["was_repaired"] else ""
            log_msg(f"🛠️ [ASTA REPAIRED ACTION] Executable Command: \"{asta_res['repaired_command']}\"{repair_str} | Mode: {route_mode}")
        else:
            log_msg(f"⚠️ [ASTA UNRECOGNIZED] Voice Input did not match supported IoT command")

    except Exception as e:
        log_msg(f"❌ [ERROR] Client {client_addr[0]}: {e}")
    finally:
        try:
            client_sock.close()
        except Exception:
            pass

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", TCP_SERVER_PORT))
    server.listen(10)

    log_msg(f"🟢 [INFINIX SERVER ONLINE] Listening on 0.0.0.0:{TCP_SERVER_PORT} for clients...")
    log_msg(f"📁 Live log file: {LOG_FILE_PATH}")

    try:
        while True:
            client_sock, client_addr = server.accept()
            t = threading.Thread(target=handle_esp32_connection, args=(client_sock, client_addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log_msg("\n[INFO] Server shutting down cleanly.")
    finally:
        server.close()

if __name__ == "__main__":
    main()
