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
            BEAM_SIZE = cfg.get("beam_size", BEAM_SIZE)
    except Exception as e:
        log_msg(f"⚠️ Failed to parse config.json: {e}")

PROTOCOL_HEARTBEAT = 0x00
PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

log_msg("=" * 60)
log_msg("🚀 [1/2] Loading Fast Offline Whisper ASR Engine...")
if HAS_FASTER_WHISPER:
    whisper_engine = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    log_msg(f"✅ Model loaded: Faster-Whisper '{WHISPER_MODEL_NAME}' ({COMPUTE_TYPE.upper()} Quantized, {CPU_THREADS} threads)")
    
    # Dummy Inference Warm-Up (Eliminates CTranslate2 First-Run Cold Start Latency Spike)
    log_msg("⚡ Pre-warming CTranslate2 CPU vector engines with 1-sec dummy inference...")
    t_warm_start = time.perf_counter()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    _ = whisper_engine.transcribe(
        dummy_audio,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        temperature=0.0,
        vad_filter=False
    )
    t_warm_end = time.perf_counter()
    log_msg(f"✅ Model Pre-Warmed in {(t_warm_end - t_warm_start)*1000:.2f} ms! Ready for sub-30ms execution.")
else:
    whisper_engine = None
    log_msg("⚠️ faster-whisper not installed. Running in high-speed simulation mode.")
log_msg("=" * 60)

def handle_esp32_connection(client_sock, client_addr):
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_sock.settimeout(60.0) # Persistent pre-warmed idle connection timeout
    is_protocol_client = False

    log_msg(f"⚡ [INCOMING CONNECT] Connection initiated from IP: {client_addr[0]}:{client_addr[1]}")

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
                log_msg(f"⚠️ [DISCONNECT] {client_addr[0]} closed persistent socket.")
                break

            opcode = opcode_byte[0]

            if opcode == PROTOCOL_HEARTBEAT:
                # 1-Byte TCP Keepalive Ping from ESP32
                continue

            if opcode == PROTOCOL_SYN:
                is_protocol_client = True
                client_sock.sendall(bytes([PROTOCOL_SYN_ACK]))
                log_msg(f"🤝 [PRE-WARMED HANDSHAKE] Binary SYN 0x01 verified. Sent SYN-ACK 0x06 to {client_addr[0]}")
                continue

            elif opcode == PROTOCOL_AUDIO_CHUNK:
                # Length-Prefixed TLV Frame: Read 2-byte N (payload length)
                len_bytes = client_sock.recv(2)
                if len(len_bytes) < 2:
                    break
                payload_len = struct.unpack("<H", len_bytes)[0]
                
                # Read N bytes of raw PCM16 audio
                chunk_data = b""
                while len(chunk_data) < payload_len:
                    more = client_sock.recv(payload_len - len(chunk_data))
                    if not more:
                        break
                    chunk_data += more
                
                pcm_chunks.append(chunk_data)

            elif opcode == PROTOCOL_STREAM_END:
                # Stream end signal received
                len_bytes = client_sock.recv(2) # Consume 2-byte length if present
                client_sock.sendall(bytes([PROTOCOL_TRANSIT_ACK]))
                log_msg(f"🚀 [TRANSIT ACK] Sent 0x7F instant hardware ACK to {client_addr[0]}")

                # Process ASR inference on ingested PCM chunks
                raw_bytes = b"".join(pcm_chunks)
                audio_dur_ms = int(len(raw_bytes) / 32)
                log_msg(f"📥 [AUDIO RECEIVED] Total bytes: {len(raw_bytes)} ({audio_dur_ms} ms) from {client_addr[0]}")

                t_asr_start = time.perf_counter()
                transcribed_text = ""
                detected_lang = "en"

                if len(raw_bytes) > 0:
                    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # 1. Noise floor check & smart peak normalization
                    max_peak = np.max(np.abs(audio_np))
                    if max_peak > 0.008:
                        audio_np = audio_np / max_peak

                    if max_peak < 0.003:
                        transcribed_text = ""
                        detected_lang = "en"
                    elif whisper_engine:
                        # 2. Hardened INT8 decoding (greedy search, no context loop, zero VAD overhead)
                        segments, info = whisper_engine.transcribe(
                            audio_np,
                            beam_size=BEAM_SIZE,
                            best_of=BEAM_SIZE,
                            condition_on_previous_text=False,
                            temperature=0.0,
                            vad_filter=False,
                            initial_prompt="English and Hindi (Latin/Hinglish) smart assistant voice commands: turn on light, fan, switch, pankha, batti, chalao, band karo, namaste, kaise ho."
                        )
                        detected_lang = getattr(info, 'language', 'en')

                        if detected_lang not in ["en", "hi"]:
                            segments, info = whisper_engine.transcribe(
                                audio_np,
                                beam_size=1,
                                language="en",
                                condition_on_previous_text=False,
                                temperature=0.0,
                                vad_filter=False,
                                initial_prompt="English and Hindi (Latin/Hinglish) smart assistant voice commands: turn on light, fan, switch, pankha, batti."
                            )
                            detected_lang = "en"

                        transcribed_text = " ".join([seg.text for seg in segments]).strip()
                    else:
                        time.sleep(0.042)
                        transcribed_text = "Sample audio processed"

                t_asr_end = time.perf_counter()
                asr_compute_ms = int((t_asr_end - t_asr_start) * 1000)

                # Send 18-Byte Telemetry Header + UTF-8 Transcribed Text Payload back to ESP32 OLED
                text_bytes = transcribed_text.encode('utf-8')
                text_len = len(text_bytes)
                try:
                    telemetry_payload = struct.pack(f"<IIIIH{text_len}s", audio_dur_ms, 0, 0, asr_compute_ms, text_len, text_bytes)
                    client_sock.sendall(telemetry_payload)
                except Exception as e:
                    log_msg(f"⚠️ Telemetry send error: {e}")

                lang_label = "Hindi 🇮🇳" if detected_lang == "hi" else ("English 🇬🇧" if detected_lang == "en" else detected_lang.upper())
                log_msg(f"✅ [STT COMPLETED] Client: {client_addr[0]} | Lang: {lang_label} | Audio: {audio_dur_ms}ms | Latency: {asr_compute_ms}ms | Text: \"{transcribed_text}\"")

                # Reset buffer for next stream on pre-warmed socket
                pcm_chunks = []

            else:
                # Raw stream fallback for non-protocol clients
                raw_data = client_sock.recv(1024)
                if not raw_data:
                    break
                pcm_chunks.append(raw_data)

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
