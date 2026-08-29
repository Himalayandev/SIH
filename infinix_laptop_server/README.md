# Infinix Laptop Offline ASR & Telemetry Server

This folder contains the offline server software to be hosted on your Infinix Laptop. It receives real-time 16kHz PCM audio streams from the ESP32 edge device, fires instant hardware transit ACKs (`0x7F`), runs Whisper ASR inference, prints the text to the laptop terminal, and sends binary telemetry structs back to the ESP32.

---

## Architecture & Communication Flow

1. **Port**: TCP `8088` (Configurable in `protocol.h`)
2. **Step 1 (Handshake)**: ESP32 sends `0x01` (SYN) $\rightarrow$ Server responds with `0x06` (SYN_ACK).
3. **Step 2 (Streaming)**: ESP32 streams raw 512-byte PCM16 chunks as the user speaks.
4. **Step 3 (Instant Transit ACK)**: The moment ESP32 sends `0xFF` (Stream-End), server replies **immediately** with `0x7F` before running inference.
5. **Step 4 (ASR & Telemetry)**: Server runs Whisper ASR, prints the transcription, and sends a 12-byte `ProfessionalTelemetry` struct to the ESP32.

---

## Running Options

### Option A: Python Fast Server (`server_faster_whisper.py`)
Quickest to run with `faster-whisper` (CTranslate2 INT8 engine):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python server_faster_whisper.py
```

### Option B: Bare-Metal C++ Server (`server_whisper.cpp`)
For maximum performance with zero runtime overhead:

```bash
# Build the binary
make

# Run the server
./whisper_server models/ggml-base.en.bin
```

---

## Performance Targets
* **Transit ACK Latency**: < 1 ms (LAN transmission time)
* **Whisper ASR Inference**: 35 – 55 ms (CPU INT8 / FP16)
* **Total Turnaround**: < 75 ms
