#!/usr/bin/env python3
"""
=============================================================================
INFINIX LAPTOP SERVER - PYTHON REAL-TIME ASR & TELEMETRY ENGINE
Uses faster-whisper (CTranslate2 INT8) with zero-delay raw TCP binary sockets
Compatible with ESP32-S3, other PCs (tcp_edge_client.py), and mobile clients.
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

# Try importing faster_whisper, else fallback to simulation
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# Load Config if present
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
    except Exception as e:
        print(f"⚠️ Failed to parse config.json: {e}")

# Protocol Constants
PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Load Whisper Model (CTranslate2 INT8 quantization on CPU)
print("=" * 60)
print("🚀 [1/2] Loading Fast Offline Whisper ASR Engine...")
if HAS_FASTER_WHISPER:
    whisper_engine = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    print(f"✅ Model loaded: Faster-Whisper '{WHISPER_MODEL_NAME}' ({COMPUTE_TYPE.upper()} Quantized, {CPU_THREADS} threads)")
else:
    whisper_engine = None
    print("⚠️ faster-whisper not installed. Running in high-speed simulation mode.")
    print("   Run: pip install faster-whisper")
print("=" * 60)

def handle_esp32_connection(client_sock, client_addr):
    # Disable Nagle's Algorithm for zero-delay writes
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_sock.settimeout(12.0)
    is_protocol_client = False

    try:
        # Read initial byte to inspect handshake
        first_byte = client_sock.recv(1)
        if not first_byte:
            client_sock.close()
            return

        pcm_chunks = []

        if first_byte[0] == PROTOCOL_SYN:
            is_protocol_client = True
            client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
            print(f"\n⚡ [STREAM INITIATED] Protocol Handshake ACK (0x06) sent to {client_addr[0]}")
        else:
            print(f"\n⚡ [STREAM INITIATED] Direct Raw Audio Client Connected ({client_addr[0]}). Receiving PCM16...")
            pcm_chunks.append(first_byte)

        # Ingest audio chunks
        while True:
            try:
                chunk = client_sock.recv(1024)
                if not chunk:
                    break

                if is_protocol_client:
                    if chunk == bytes([PROTOCOL_STREAM_END]):
                        client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                        print("🚀 [TRANSIT ACK] 0x7F sent to client instantly.")
                        break
                    elif len(chunk) > 1 and chunk[-1] == PROTOCOL_STREAM_END:
                        pcm_chunks.append(chunk[:-1])
                        client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                        print("🚀 [TRANSIT ACK] 0x7F sent to client instantly.")
                        break

                pcm_chunks.append(chunk)
            except socket.timeout:
                break
            except Exception:
                break

        raw_bytes = b"".join(pcm_chunks)
        audio_dur_ms = int(len(raw_bytes) / 32)  # 16000 samples/sec * 2 bytes = 32 bytes/ms

        # Transcribe using Whisper (Hindi & English Multilingual Auto-Detect)
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
                transcribed_text = "activate turn on the lights"

        t_asr_end = time.perf_counter()
        asr_compute_ms = int((t_asr_end - t_asr_start) * 1000)

        # Send 16-byte telemetry struct if protocol client
        if is_protocol_client:
            try:
                telemetry_payload = struct.pack("<IIII", audio_dur_ms, 0, 0, asr_compute_ms)
                client_sock.sendall(telemetry_payload)
            except Exception:
                pass

        lang_label = "Hindi 🇮🇳" if detected_lang == "hi" else ("English 🇬🇧" if detected_lang == "en" else detected_lang.upper())

        print("\n" + "┌" + "─" * 56 + "┐")
        print("│            📊 PROFESSIONAL METRICS LOG (SERVER)           │")
        print("├" + "─" * 56 + "┤")
        print(f"│ • Client Address     : {client_addr[0]:<34} │")
        print(f"│ • Language Detected  : {lang_label:<34} │")
        print(f"│ • Audio Duration     : {audio_dur_ms:6d} ms                           │")
        print(f"│ • ASR Inference Time : {asr_compute_ms:6d} ms                           │")
        print(f"│ • Live Transcription : \"{transcribed_text[:30]:<30}\" │")
        print("└" + "─" * 56 + "┘\n")

    except Exception as e:
        print(f"[-] Error processing stream from {client_addr[0]}: {e}")
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

    print(f"🟢 [INFINIX SERVER ONLINE] Listening on 0.0.0.0:{TCP_SERVER_PORT} for clients...")

    try:
        while True:
            client_sock, client_addr = server.accept()
            t = threading.Thread(target=handle_esp32_connection, args=(client_sock, client_addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down cleanly.")
    finally:
        server.close()

if __name__ == "__main__":
    main()
