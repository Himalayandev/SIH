# ESP32-S3 Dual-Core Edge Firmware

Engineered specifically for the **ESP32-S3 (240MHz, 8MB Flash, 8MB PSRAM)** with vector SIMD acceleration, **INMP441 I2S MEMS Microphone**, and **TekBud 0.96-inch SSD1306 OLED**.

---

## 📌 Verified Hardware Pin Mapping

```
[ INMP441 I2S Microphone ]                 [ ESP32-S3 Dev Board ]                 [ 0.96" OLED Display ]
┌──────────────────────────┐               ┌──────────────────────┐               ┌──────────────────────┐
│  VDD ────────────────────┼──────────────►│ 3V3                  │◄──────────────┼───── VCC             │
│  GND ────────────────────┼──────────────►│ GND                  │◄──────────────┼───── GND             │
│  L/R (Left/Right Select)─┼──► (To GND)   │                      │               │                      │
│  SCK (Serial Clock) ─────┼──────────────►│ GPIO 12 (I2S SCK)    │               │                      │
│  WS (Word Select) ───────┼──────────────►│ GPIO 13 (I2S WS)     │               │                      │
│  SD (Serial Data) ───────┼──────────────►│ GPIO 11 (I2S SD)     │               │                      │
│                          │               │                      │               │                      │
│                          │               │ GPIO 4 (I2C SDA) ────┼──────────────►│ SDA                  │
│                          │               │ GPIO 5 (I2C SCL) ────┼──────────────►│ SCL                  │
└──────────────────────────┘               └──────────────────────┘               └──────────────────────┘
```

> **Note on L/R pin**: Solder or connect the `L/R` pin directly to **GND** on the INMP441. This forces single-channel left audio output for 16-bit mono 16kHz audio.

---

## ⚡ Multi-Core Optimization Details

1. **Core 0 (Protocol CPU)**:
   - Dedicated TCP Socket Pipeline with `TCP_NODELAY`.
   - Sends the 1-byte handshake `SYN` (`0x01`) and streams 512-byte PCM chunks.
   - Captures instant hardware `0x7F` transit ACK from the laptop.

2. **Core 1 (Application / DSP CPU)**:
   - FreeRTOS I2S DMA Ring Buffer (6 DMA buffers of 256 samples).
   - Energy VAD silence detection and TFLite Micro INT8 inference using ESP32-S3 Vector Extensions.
   - Pushes raw audio chunks into a FreeRTOS lock-free Queue.

3. **High-Speed I2C Bus**:
   - Clock set to **400 kHz** (`Wire.setClock(400000)`), ensuring screen updates never stall the audio DMA pipeline.

---

## 🛠️ Build & Flash

```bash
cd esp32_edge_firmware
pio run -t upload -t monitor
```
