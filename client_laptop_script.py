#!/usr/bin/env python3
"""
=============================================================================
CLIENT LAPTOP STREAMER SCRIPT (Run on Laptop 2)
Connects to Laptop 1 (new_server.py) over Wi-Fi / Local Network.
Streams live microphone PCM audio or synthetic speech and prints real-time latency.
=============================================================================
"""

import sys
import time
import socket
import struct
import argparse
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

def stream_mic_or_tone(server_ip, server_port=8088, duration=3.0, use_mic=True):
    print("=" * 65)
    print(f"📡 [CLIENT LAPTOP 2] Connecting to STT Server on {server_ip}:{server_port}...")
    print("=" * 65)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10.0)

        t0 = time.perf_counter()
        sock.connect((server_ip, server_port))
        t_conn = (time.perf_counter() - t0) * 1000
        print(f"✅ Connected in {t_conn:.1f} ms!")

        # 1. SYN Handshake
        sock.sendall(bytes([PROTOCOL_SYN]))
        ack = sock.recv(1)
        if not ack or ack[0] != PROTOCOL_SYN_ACK:
            print(f"❌ Handshake failed! Received: {ack}")
            return

        print("🤝 Handshake SYN-ACK (0x06) verified!")

        # 2. Capture PCM Audio
        sample_rate = 16000
        if use_mic and HAS_SOUNDDEVICE:
            print(f"🎙️ Recording live microphone for {duration} seconds... Speak now!")
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            pcm_bytes = audio_data.tobytes()
        else:
            print(f"🔊 Generating {duration}s synthetic audio tone...")
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
            pcm_bytes = tone.tobytes()

        # 3. Stream PCM Audio Chunks (512 bytes per chunk)
        print(f"🚀 Streaming {len(pcm_bytes)} bytes audio over TCP network...")
        t_stream_start = time.perf_counter()

        chunk_size = 512
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]
            header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(chunk))
            sock.sendall(header + chunk)
            time.sleep(0.004)  # Simulate real-time audio chunk intervals

        # 4. Stream End (0xFF)
        sock.sendall(struct.pack("<BH", PROTOCOL_STREAM_END, 0))
        t_stream_end = time.perf_counter()

        # 5. Read Instant Transit ACK (0x7F)
        transit_ack = sock.recv(1)
        if transit_ack and transit_ack[0] == PROTOCOL_TRANSIT_ACK:
            ack_delay = (time.perf_counter() - t_stream_end) * 1000
            print(f"⚡ [TRANSIT ACK 0x7F] Server acknowledged in {ack_delay:.1f} ms!")

        # 6. Read 18-Byte Telemetry Header + UTF-8 STT Output
        telemetry = sock.recv(18)
        if len(telemetry) == 18:
            audio_dur, edge_ms, transfer_ms, asr_ms, text_len = struct.unpack("<IIIIH", telemetry)
            text_str = ""
            if text_len > 0:
                text_bytes = sock.recv(text_len)
                text_str = text_bytes.decode('utf-8', errors='ignore')

            print("\n" + "=" * 65)
            print("📊 LATENCY METRICS RECEIVED FROM SERVER:")
            print(f"   ⏱️ Data Transfer Time (Client -> Server): {transfer_ms} ms")
            print(f"   ⚡ Server STT Compute Latency:           {asr_ms} ms")
            print(f"   🚀 Total End-to-End Latency:             {transfer_ms + asr_ms} ms")
            print(f"   🗣️ Transcribed STT Text:                \"{text_str}\"")
            print("=" * 65 + "\n")

        sock.close()

    except Exception as e:
        print(f"❌ Connection error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Laptop 2 STT Client Streamer")
    parser.add_argument("--host", required=True, help="IP address of Laptop 1 running new_server.py")
    parser.add_argument("--port", type=int, default=8088, help="Server Port (default: 8088)")
    parser.add_argument("--duration", type=float, default=3.0, help="Speech recording duration in seconds")
    parser.add_argument("--synth", action="store_true", help="Use synthetic tone instead of microphone")
    args = parser.parse_args()

    stream_mic_or_tone(args.host, args.port, args.duration, use_mic=not args.synth)

if __name__ == "__main__":
    main()
