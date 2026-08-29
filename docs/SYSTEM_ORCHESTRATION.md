# 🔄 System Orchestration & User Operational Guide

This document details the end-to-end operational orchestration of the **ESP32-S3 Edge Device** and the **Infinix Laptop Server**. Follow this sequence to assemble, pair, and run the system seamlessly.

---

## 🗺️ 1. Physical Hardware Setup

```
 [ INMP441 I2S Mic ]                 [ ESP32-S3 Dev Board ]                 [ 0.96" OLED Screen ]
 ┌──────────────────┐               ┌──────────────────────┐               ┌───────────────────┐
 │ VDD ─────────────┼──────────────►│ 3V3                  │◄──────────────┼──── VCC           │
 │ GND ─────────────┼──────────────►│ GND                  │◄──────────────┼──── GND           │
 │ L/R ─────────────┼──► (To GND)   │                      │               │                   │
 │ SCK ─────────────┼──────────────►│ GPIO 12              │               │                   │
 │ WS  ─────────────┼──────────────►│ GPIO 13              │               │                   │
 │ SD  ─────────────┼──────────────►│ GPIO 11              │               │                   │
 │                  │               │ GPIO 4 (SDA) ────────┼──────────────►│ SDA               │
 │                  │               │ GPIO 5 (SCL) ────────┼──────────────►│ SCL               │
 └──────────────────┘               └──────────────────────┘               └───────────────────┘
```

> ⚠️ **Critical Check**: Verify that `L/R` on the INMP441 is connected to `GND`. This configures left-channel mono audio at 16kHz.

---

## 🌐 2. Network & Hotspot Pairing

For zero jitter, connect both the Infinix Laptop and the ESP32-S3 to the same WiFi network or your phone's Wi-Fi hotspot:

1. **Find Laptop IP**:
   * **Windows**: Run `ipconfig` $\rightarrow$ Look for `IPv4 Address` under Wireless LAN adapter (e.g. `192.168.1.100`).
   * **Linux / Mac**: Run `ifconfig` or `ip a`.
2. **Update ESP32 Firmware Settings**:
   Open [`esp32_edge_firmware/main.cpp`](file:///Users/namanbhatt/.gemini/antigravity-ide/scratch/SIH/esp32_edge_firmware/main.cpp) (Lines 42-44):
   ```cpp
   const char* WIFI_SSID = "Your_Hotspot_Name";
   const char* WIFI_PASS = "Your_Password";
   const char* SERVER_IP = "192.168.1.100"; // Put your Laptop IP here
   ```

---

## 🚀 3. Step-by-Step Execution Sequence

### Phase 1: Launch Laptop Server First
On the Infinix laptop, open a terminal in the project directory:

```bash
cd infinix_laptop_server

# Option A: Bare-Metal C++ Engine (Recommended for judges)
make
./whisper_server

# Option B: Python Real-Time Engine (Instant)
python server_faster_whisper.py
```
*Expected Console Output*:
```text
🟢 [INFINIX SERVER ONLINE] Listening on port 8088 for ESP32 streams...
```

---

### Phase 2: Power Up the ESP32-S3 Edge Device
1. Connect the ESP32-S3 via USB-C.
2. Flash the firmware using PlatformIO:
   ```bash
   cd esp32_edge_firmware
   pio run -t upload -t monitor
   ```
3. *Expected OLED Display*:
   ```text
   ┌────────────────────┐
   │ 🟢 SYSTEM READY    │
   │ Idle Listening...  │
   │ Say 'Activate'     │
   └────────────────────┘
   ```

---

### Phase 3: Live Voice Command Interaction
1. **Speak Wake-Word**: Say `"Activate"` near the INMP441 microphone.
2. **Instant Visual Feedback**:
   * ESP32 OLED immediately turns to: `🎙️ STREAMING LIVE... >> Speak command <<`
   * Laptop prints: `⚡ [STREAM INITIATED] Handshake ACK sent.`
3. **Speak Command**: Say `"Turn on the kitchen lights"` or `"Increase bedroom temperature"`.
4. **Pause (Stop Speaking)**:
   * Edge VAD detects trailing silence (1.2s).
   * ESP32 flushes stream-end byte (`0xFF`).
   * Laptop fires instant hardware ACK (`0x7F`).
   * Whisper decodes text.
5. **Dashboard Updates Simultaneously**:
   * **Laptop Terminal**: Prints the transcribed text + compute performance card.
   * **ESP32 OLED**: Prints the full millisecond latency profile:
     ```text
     ┌───────────────────────────┐
     │ SYSTEM LATENCY (LIVE)     │
     │ 1. Voice->Net : 12 ms     │
     │ 2. Net ACK RTT:  3 ms     │
     │ 3. Stream Dur : 2840 ms   │
     │ 4. Server ASR : 42 ms     │
     │ STATUS: SERVER ACK ✅     │
     └───────────────────────────┘
     ```

---

## 🧪 4. Automated Testing / Simulation (No Hardware Required)

You can verify the entire server pipeline without having the physical ESP32 attached:

```bash
python3 tests/test_end_to_end.py
```
This script acts as a virtual ESP32-S3, verifying TCP handshake, PCM stream ingestion, sub-millisecond ACK turnaround, and binary telemetry decoding.
