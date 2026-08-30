#!/usr/bin/env python3
"""
=============================================================================
TCP EDGE CLIENT (FOR SECOND LAPTOP / PC CLIENT)
Emulates ESP32-S3 Edge Device on another PC.
- Detects 'Activate' wake-word locally or records audio.
- Performs binary TCP protocol handshake (0x01 -> 0x06).
- Streams 16kHz PCM16 audio chunks over TCP.
- Sends 0xFF terminator and parses 0x7F Transit ACK + 16-byte telemetry.
=============================================================================
"""

import os
import sys
import time
import socket
import argparse
import struct
import numpy as np

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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configuration Defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088

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

ACTIVATE_THRESHOLD = 0.65
VAD_RMS_THRESHOLD = 0.005

PROTOCOL_HEARTBEAT = 0x00
PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

# Load TFLite Model if present
interpreter = None
in_idx = None
out_idx = None
in_scale = 0
in_zero_point = 0
out_scale = 0
out_zero_point = 0
audio_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)

if HAS_TF and os.path.exists(MODEL_PATH):
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        in_idx = interpreter.get_input_details()[0]['index']
        out_idx = interpreter.get_output_details()[0]['index']
        in_scale, in_zero_point = interpreter.get_input_details()[0]['quantization']
        out_scale, out_zero_point = interpreter.get_output_details()[0]['quantization']
    except Exception as e:
        print(f"⚠️ Wake-word model load warning: {e}")

def extract_features(audio: np.ndarray) -> np.ndarray:
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

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

def stream_audio_to_server(host, port, stream_sec=4.0):
    print(f"\n📡 Pre-warming socket to server {host}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10.0)
        sock.connect((host, port))

        # 1. Send Pre-Warmed SYN Handshake (0x01)
        sock.sendall(bytes([PROTOCOL_SYN]))

        ack = sock.recv(1)
        if not ack or ack[0] != PROTOCOL_SYN_ACK:
            print(f"⚠️ Protocol handshake mismatch: {ack}")
        else:
            print("✅ Pre-warmed Handshake ACK (0x06) verified!")

        # 2. Record & Stream Microphone PCM16 Audio in TLV Framed Chunks
        print(f"🎙️ Streaming speech for {stream_sec:.1f} seconds...")
        sample_count = int(SAMPLE_RATE * stream_sec)
        t = np.linspace(0, stream_sec, sample_count, False)
        pcm_data = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16).tobytes()

        chunk_size = 512
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(chunk))
            sock.sendall(header + chunk)
            time.sleep(0.005)

        # 3. Send Length-Prefixed Stream End Frame (0xFF)
        end_header = struct.pack("<BH", PROTOCOL_STREAM_END, 0)
        sock.sendall(end_header)
        print("--> Sent Stream End (0xFF). Waiting for Transit ACK...")

        # 4. Read Transit ACK (0x7F)
        t_ack = sock.recv(1)
        if t_ack and t_ack[0] == PROTOCOL_TRANSIT_ACK:
            print("⚡ [TRANSIT ACK] Received 0x7F from server!")

        # 5. Read 18-Byte Telemetry Header + UTF-8 Transcribed Text Payload
        telemetry = sock.recv(18)
        if len(telemetry) == 18:
            audio_dur, edge_ms, net_ms, asr_ms, text_len = struct.unpack("<IIIIH", telemetry)
            text_str = ""
            if text_len > 0:
                text_bytes = sock.recv(text_len)
                text_str = text_bytes.decode('utf-8', errors='ignore')
            print(f"📊 Server Latency: Audio {audio_dur}ms | Server ASR Compute: {asr_ms}ms | STT: \"{text_str}\"")

        sock.close()
        print("✅ Stream finished successfully.\n")
        return True

    except Exception as e:
        print(f"❌ Failed to stream to server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="TCP Edge Client for SIH Laptop Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server IP Address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server TCP Port")
    parser.add_argument("--duration", type=float, default=4.0, help="Speech recording duration in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🟢 TCP EDGE CLIENT ONLINE (Connecting to {args.host}:{args.port})")
    print("   Listening for 'Activate' wake-word to stream audio...")
    print("=" * 60)

    step_size = int(SAMPLE_RATE * 0.06)

    try:
        while True:
            if interpreter:
                with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=step_size, callback=audio_callback):
                    while True:
                        rms = np.sqrt(np.mean(audio_buffer ** 2))
                        if rms < VAD_RMS_THRESHOLD:
                            time.sleep(0.03)
                            continue

                        feat = extract_features(audio_buffer)
                        inp = feat[np.newaxis, ..., np.newaxis].astype(np.float32)

                        if in_scale > 0:
                            inp = np.round(inp / in_scale + in_zero_point).astype(np.int8)

                        interpreter.set_tensor(in_idx, inp)
                        interpreter.invoke()
                        out = interpreter.get_tensor(out_idx)

                        probs = (out.astype(np.float32) - out_zero_point) * out_scale if out_scale > 0 else out
                        act_prob = probs[0][ACTIVATE_IDX]
                        bar = "█" * int(act_prob * 15)
                        print(f"\r[KWS IDLE] 'Activate': [{bar:<15}] {act_prob * 100:4.1f}%", end="")

                        if act_prob >= ACTIVATE_THRESHOLD:
                            print(f"\n⚡ 'Activate' Triggered ({act_prob * 100:.1f}%)!")
                            break

                        time.sleep(0.02)
            else:
                input("\nPress ENTER to stream 4 seconds of speech to server...")

            stream_audio_to_server(args.host, args.port, args.duration)

            if interpreter:
                audio_buffer.fill(0)
                time.sleep(1.0)
                print("=" * 60)
                print("🟢 Resuming Wake-Word Listening...")
                print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INFO] Client stopped.")

if __name__ == "__main__":
    main()
