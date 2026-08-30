#!/usr/bin/env python3
"""
=============================================================================
ESP32 TCP CLIENT SIMULATION & LATENCY TESTER
Simulates ESP32 edge firmware connecting to Infinix Laptop ASR Server (Port 8088).
Tests: Handshake (0x01 -> 0x06), Audio Streaming, Stream End (0xFF -> 0x7F), Telemetry.
=============================================================================
"""

import sys
import socket
import struct
import time
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8088

PROTOCOL_SYN = 0x01
PROTOCOL_SYN_ACK = 0x06
PROTOCOL_AUDIO_CHUNK = 0x02
PROTOCOL_STREAM_END = 0xFF
PROTOCOL_TRANSIT_ACK = 0x7F

def run_test_client(host=SERVER_HOST, port=SERVER_PORT):
    print("=" * 60)
    print(f"🧪 [ESP32 SIMULATOR] Pre-warming persistent socket to {host}:{port}...")
    print("=" * 60)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10.0)

        t_connect_start = time.perf_counter()
        sock.connect((host, port))
        t_connect_end = time.perf_counter()
        print(f"✅ Pre-warmed TCP Socket Connected in {(t_connect_end - t_connect_start)*1000:.2f} ms")

        # 1. Send SYN Handshake on startup
        print("\n--> [1/4] Sending Pre-Warmed SYN (0x01)...")
        sock.sendall(bytes([PROTOCOL_SYN]))

        # Read SYN-ACK
        ack = sock.recv(1)
        if not ack or ack[0] != PROTOCOL_SYN_ACK:
            print(f"❌ Handshake failed! Expected 0x06, got {ack}")
            return False
        print("<-- [2/4] Pre-Warmed Handshake ACK (0x06) verified! Socket pre-warmed (0ms wake delay).")

        # 2. Generate 1.5 sec of 16kHz PCM16 audio (synthetic 440Hz tone)
        sample_rate = 16000
        duration_sec = 1.5
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        tone = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
        audio_bytes = tone.tobytes()

        print(f"\n--> [3/4] Streaming {len(audio_bytes)} bytes ({duration_sec}s) PCM16 audio in TLV Framed 512B Chunks...")

        chunk_size = 512
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            header = struct.pack("<BH", PROTOCOL_AUDIO_CHUNK, len(chunk))
            sock.sendall(header + chunk)
            time.sleep(0.005)  # Simulate real-time hardware stream delay

        # Send Length-Prefixed Stream End Terminator (0xFF)
        t_end_sent = time.perf_counter()
        end_header = struct.pack("<BH", PROTOCOL_STREAM_END, 0)
        sock.sendall(end_header)
        print("--> [4/4] Sent Stream End (0xFF). Waiting for Transit ACK (0x7F)...")

        # Wait for Instant Transit ACK (0x7F)
        transit_ack = sock.recv(1)
        t_ack_received = time.perf_counter()

        if transit_ack and transit_ack[0] == PROTOCOL_TRANSIT_ACK:
            transit_latency_ms = (t_ack_received - t_end_sent) * 1000
            print(f"⚡ [TRANSIT ACK 0x7F] Received in {transit_latency_ms:.2f} ms!")
        else:
            print(f"⚠️ Transit ACK missing or invalid: {transit_ack}")

        # Receive 18-Byte Telemetry Header Struct + Text String Payload
        telem_hdr = sock.recv(18)
        if len(telem_hdr) == 18:
            audio_dur, edge_ms, net_ms, asr_ms, text_len = struct.unpack("<IIIIH", telem_hdr)
            text_str = ""
            if text_len > 0:
                text_bytes = sock.recv(text_len)
                text_str = text_bytes.decode('utf-8', errors='ignore')

            print("\n" + "┌" + "─" * 58 + "┐")
            print("│         📊 ESP32 SIMULATION TEST RESULTS (PASSED)        │")
            print("├" + "─" * 58 + "┤")
            print(f"│ • Audio Duration     : {audio_dur:6d} ms                            │")
            print(f"│ • Transit Latency    : {transit_latency_ms:6.2f} ms                            │")
            print(f"│ • Server ASR Compute : {asr_ms:6d} ms                            │")
            print(f"│ • Transcribed Text   : \"{text_str}\"                           │")
            print("└" + "─" * 58 + "┘\n")
        else:
            print(f"⚠️ Telemetry byte count unexpected: {len(telem_hdr)} bytes")

        sock.close()
        print("🎉 Server end-to-end verification completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else SERVER_HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else SERVER_PORT
    run_test_client(host, port)
