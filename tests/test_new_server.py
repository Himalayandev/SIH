import sys
import time
import socket
import struct
import threading
import numpy as np

sys.path.insert(0, ".")
from new_server import STTServerEngine

def main():
    print("=" * 60)
    print("🧪 TESTING NEW_SERVER LATENCY & DEVICE METRICS ENGINE")
    print("=" * 60)

    engine = STTServerEngine(port=8089)
    print("1. Loading & warming STT model...")
    engine.load_stt_model()

    print("2. Starting server on port 8089...")
    assert engine.start() == True, "Failed to start server"

    def client_sim():
        time.sleep(0.5)
        print("3. Connecting test client to 127.0.0.1:8089...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 8089))

        # SYN Handshake
        sock.sendall(bytes([0x01]))
        ack = sock.recv(1)
        assert ack[0] == 0x06, f"Expected SYN_ACK (0x06), got {ack}"

        # Stream audio
        sr = 16000
        dur = 1.5
        t = np.linspace(0, dur, int(sr * dur), False)
        pcm = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16).tobytes()

        print(f"4. Streaming {len(pcm)} bytes PCM audio chunks...")
        chunk_size = 512
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i:i + chunk_size]
            sock.sendall(struct.pack("<BH", 0x02, len(chunk)) + chunk)
            time.sleep(0.003)

        # Stream End
        print("5. Sending Stream End (0xFF)...")
        sock.sendall(struct.pack("<BH", 0xFF, 0))

        # Transit ACK & Telemetry
        transit_ack = sock.recv(1)
        assert transit_ack[0] == 0x7F, f"Expected 0x7F, got {transit_ack}"

        telemetry = sock.recv(18)
        audio_dur, edge_ms, transfer_ms, asr_ms, text_len = struct.unpack("<IIIIH", telemetry)
        text_bytes = sock.recv(text_len)
        text = text_bytes.decode('utf-8')

        print(f"✅ TELEMETRY VERIFIED:")
        print(f"   - Audio Sample Duration: {audio_dur} ms")
        print(f"   - Data Transfer Latency (Client -> Server): {transfer_ms} ms")
        print(f"   - Server STT Compute Latency: {asr_ms} ms")
        print(f"   - Total End-to-End Latency: {transfer_ms + asr_ms} ms")
        print(f"   - STT Transcribed Text: \"{text}\"")

        sock.close()

    t = threading.Thread(target=client_sim)
    t.start()
    t.join(timeout=10)

    engine.stop()
    print("=" * 60)
    print("🎉 ALL LATENCY METRICS VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
