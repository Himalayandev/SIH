#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

// Network Configuration
#define TCP_SERVER_PORT      8088

// Binary Protocol Command Bytes
#define PROTOCOL_SYN         0x01 // ESP32 -> Server: Initiation request
#define PROTOCOL_SYN_ACK     0x06 // Server -> ESP32: Handshake acknowledged
#define PROTOCOL_STREAM_END  0xFF // ESP32 -> Server: Silence cut-off / end of stream
#define PROTOCOL_TRANSIT_ACK 0x7F // Server -> ESP32: Instant receipt ACK before ASR compute

// 12-Byte Packed Telemetry Struct (Zero byte-padding overhead)
#pragma pack(push, 1)
struct ProfessionalTelemetry {
    uint32_t audio_duration_ms;     // Total duration of captured voice sample (ms)
    uint32_t edge_processing_ms;    // Δt1: Silence detection & DMA flush latency on ESP32 (ms)
    uint32_t network_transit_ms;    // Δt2 + Δt3: Network ACK round-trip ping time (ms)
    uint32_t server_asr_compute_ms; // Server-side Whisper ASR inference latency (ms)
};
#pragma pack(pop)

#endif // PROTOCOL_H
