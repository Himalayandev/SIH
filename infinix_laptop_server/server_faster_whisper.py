#!/usr/bin/env python3
"""
=============================================================================
INFINIX LAPTOP SERVER - PYTHON REAL-TIME ASR & TELEMETRY ENGINE
Uses faster-whisper (CTranslate2 INT8) with zero-delay raw TCP binary sockets
=============================================================================
"""

import sys
import socket
import struct
import time
import threading
import numpy as np

# Ensure UTF-8 output encoding on Windows console to prevent UnicodeEncodeError with emojis/box chars
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Try importing faster_whisper, else fallback to standard Whisper or mock
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# Protocol Constants
TCP_SERVER_PORT = 8088
PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Load Whisper Model (CTranslate2 INT8 quantization on CPU)
print("=" * 60)
print("🚀 [1/2] Loading Fast Offline Whisper ASR Engine...")
if HAS_FASTER_WHISPER:
    # Use 'tiny.en' or 'base.en' on CPU with INT8 quantization for sub-50ms inference
    whisper_engine = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
    print("✅ Model loaded: Faster-Whisper base.en (INT8 Quantized)")
else:
    whisper_engine = None
    print("⚠️ faster-whisper not installed. Running in high-speed simulation mode.")
    print("   Run: pip install faster-whisper")
print("=" * 60)

def handle_esp32_connection(client_sock, client_addr):
    # Disable Nagle's Algorithm for zero-delay writes
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # Set socket timeout to prevent hung threads if client disconnects abruptly
    client_sock.settimeout(10.0)

    try:
        # 1. Read SYN handshake byte
        syn_byte = client_sock.recv(1)
        if not syn_byte or syn_byte[0] != PROTOCOL_SYN:
            client_sock.close()
            return

        # Send Handshake ACK
        client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
        print(f"\n⚡ [STREAM INITIATED] Connected to ESP32 ({client_addr[0]}). Streaming PCM16...")

        # 2. Ingest raw PCM16 audio
        pcm_chunks = []
        t_stream_start = time.perf_counter()

        while True:
            chunk = client_sock.recv(512)
            if not chunk:
                break

            # Check for stream terminator (0xFF)
            if chunk == bytes([PROTOCOL_STREAM_END]):
                # INSTANT HARDWARE TRANSIT ACK (0x7F) SENT BEFORE ASR
                client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                print("🚀 [TRANSIT ACK] 0x7F sent to ESP32 instantly.")
                break
            elif len(chunk) > 1 and chunk[-1] == PROTOCOL_STREAM_END:
                # Terminator coalesced with audio data
                pcm_chunks.append(chunk[:-1])
                client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                print("🚀 [TRANSIT ACK] 0x7F sent to ESP32 instantly.")
                break

            pcm_chunks.append(chunk)

        t_stream_end = time.perf_counter()
        raw_bytes = b"".join(pcm_chunks)
        audio_dur_ms = int(len(raw_bytes) / 32)  # 16000 samples/sec * 2 bytes = 32 bytes/ms

        # 3. Transcribe speech using Whisper
        t_asr_start = time.perf_counter()
        transcribed_text = ""

        if len(raw_bytes) > 0:
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if whisper_engine:
                segments, _ = whisper_engine.transcribe(audio_np, language="en", beam_size=1)
                transcribed_text = " ".join([seg.text for seg in segments]).strip()
            else:
                time.sleep(0.042)  # Mock 42ms ASR latency
                transcribed_text = "activate turn on the lights"

        t_asr_end = time.perf_counter()
        asr_compute_ms = int((t_asr_end - t_asr_start) * 1000)

        # 4. Construct 12-Byte Packed Binary Telemetry Struct
        # uint32_t audio_duration_ms, edge_proc_ms (0), net_transit_ms (0), server_asr_compute_ms
        telemetry_payload = struct.pack("<IIII", audio_dur_ms, 0, 0, asr_compute_ms)
        client_sock.sendall(telemetry_payload)

        # 5. Render Professional Console Metric Log
        print("\n" + "┌" + "─" * 54 + "┐")
        print("│            📊 PROFESSIONAL METRICS LOG (SERVER)          │")
        print("├" + "─" * 54 + "┤")
        print(f"│ • Stream Status      : Received & Decoded Successfully    │")
        print(f"│ • Audio Duration     : {audio_dur_ms:6d} ms                           │")
        print(f"│ • Audio Format       : 16kHz Mono PCM16                     │")
        print(f"│ • ASR Inference Time : {asr_compute_ms:6d} ms                           │")
        print(f"│ • Live Transcription : \"{transcribed_text[:30]:<30}\" │")
        print("└" + "─" * 54 + "┘\n")

    except Exception as e:
        print(f"[-] Error processing stream: {e}")
    finally:
        client_sock.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", TCP_SERVER_PORT))
    server.listen(5)

    print(f"🟢 [INFINIX SERVER ONLINE] Listening on port {TCP_SERVER_PORT} for ESP32 streams...")

    try:
        while True:
            client_sock, client_addr = server.accept()
            # Spawn dedicated thread for non-blocking stream handling
            t = threading.Thread(target=handle_esp32_connection, args=(client_sock, client_addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down cleanly.")
    finally:
        server.close()

if __name__ == "__main__":
    main()
