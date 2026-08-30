#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

// Network Configuration
#define TCP_SERVER_PORT      8088

// Binary Protocol Opcodes
#define PROTOCOL_HEARTBEAT   0x00 // ESP32 -> Server: Idle TCP keepalive ping (1-byte)
#define PROTOCOL_SYN         0x01 // ESP32 -> Server: Connection handshake request
#define PROTOCOL_SYN_ACK     0x06 // Server -> ESP32: Handshake acknowledged
#define PROTOCOL_AUDIO_CHUNK 0x02 // ESP32 -> Server: Length-prefixed audio chunk packet
#define PROTOCOL_STREAM_END  0xFF // ESP32 -> Server: Length-prefixed stream end packet
#define PROTOCOL_TRANSIT_ACK 0x7F // Server -> ESP32: Instant receipt ACK before ASR compute

// Length-Prefixed TLV Frame Header (Prevents 0xFF byte collision in raw PCM16)
#pragma pack(push, 1)
struct ChunkHeader {
    uint8_t opcode;     // 0x02 for AUDIO_CHUNK | 0xFF for STREAM_END
    uint16_t length;    // Payload length N bytes (little-endian)
};

// 18-Byte Packed Telemetry Struct + UTF-8 Text String Payload
struct ProfessionalTelemetry {
    uint32_t audio_duration_ms;     // Total duration of captured voice sample (ms)
    uint32_t edge_processing_ms;    // Δt1: Silence detection & DMA flush latency on ESP32 (ms)
    uint32_t network_transit_ms;    // Δt2 + Δt3: Network ACK round-trip ping time (ms)
    uint32_t server_asr_compute_ms; // Server-side Whisper ASR inference latency (ms)
    uint16_t text_length;           // UTF-8 transcribed speech text payload length in bytes
};
#pragma pack(pop)

#endif // PROTOCOL_H
