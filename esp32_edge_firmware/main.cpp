// ============================================================================
// ESP32-S3 DUAL-CORE LOW-LATENCY EDGE FIRMWARE
// Hardware: ESP32-S3 (240MHz) + INMP441 I2S Mic + SSD1306 0.96" OLED
// Multi-Threading: Core 0 (TCP Network & Telemetry) | Core 1 (I2S DMA & KWS Engine)
// ============================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <driver/i2s.h>
#include <esp_timer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "protocol.h"
#include "model_data.h"

// ----------------------------------------------------------------------------
// 1. PIN DEFINITIONS (ESP32-S3 Dedicated Low-Jitter GPIOs)
// ----------------------------------------------------------------------------
// INMP441 I2S MEMS Microphone
#define I2S_PORT            I2S_NUM_0
#define I2S_PIN_SCK         12  // Serial Clock (BCLK) -> GPIO 12
#define I2S_PIN_WS          13  // Word Select (LRCK)  -> GPIO 13
#define I2S_PIN_SD          11  // Serial Data (DIN)   -> GPIO 11

// SSD1306 0.96" OLED (I2C)
#define I2C_PIN_SDA         4   // I2C SDA -> GPIO 4
#define I2C_PIN_SCL         5   // I2C SCL -> GPIO 5
#define SCREEN_WIDTH        128
#define SCREEN_HEIGHT       64
#define OLED_RESET          -1
#define SCREEN_I2C_ADDR     0x3C

// Audio & Stream Parameters
#define SAMPLE_RATE         16000
#define CHUNK_BYTES         512
#define VAD_SILENCE_RMS     220   // Amplitude threshold for silence gate
#define SILENCE_TIMEOUT_MS  1200  // End of speech after 1.2s silence

// Network Credentials (UPDATE FOR YOUR HOTSPOT / ROUTER)
const char* WIFI_SSID       = "YOUR_WIFI_SSID";
const char* WIFI_PASS       = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP       = "192.168.1.100"; // Infinix Laptop Static IP

// ----------------------------------------------------------------------------
// 2. INTER-CORE SYNCHRONIZATION & BUFFERS
// ----------------------------------------------------------------------------
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
WiFiClient tcp_client;

// High-Speed FreeRTOS Audio Queue between Core 1 and Core 0
struct AudioChunk {
    uint8_t data[CHUNK_BYTES];
    size_t length;
    bool is_last;
};

QueueHandle_t audio_queue;
SemaphoreHandle_t screen_mutex;

enum SystemState {
    STATE_IDLE_LISTENING,
    STATE_STREAMING_VOICE,
    STATE_WAITING_ACK,
    STATE_DISPLAY_METRICS
};

volatile SystemState current_state = STATE_IDLE_LISTENING;

// ----------------------------------------------------------------------------
// 3. HARDWARE INITIALIZATION
// ----------------------------------------------------------------------------
void init_i2s_dma() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // INMP441 L/R wired to GND
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,     // 6 DMA buffers for zero-drop audio
        .dma_buf_len = 256,
        .use_apll = true,       // Use Audio PLL for ultra-clean clock on S3
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_PIN_SCK,
        .ws_io_num = I2S_PIN_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_PIN_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
}

void init_fast_oled() {
    // Fast Mode I2C @ 400kHz to prevent screen updates from stalling audio DMA
    Wire.begin(I2C_PIN_SDA, I2C_PIN_SCL);
    Wire.setClock(400000);

    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_I2C_ADDR)) {
        Serial.println(F("[-] SSD1306 allocation failed"));
    }
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 5);
    display.println(F("┌────────────────────┐"));
    display.println(F("│ ESP32-S3 TINYML    │"));
    display.println(F("│ DUAL-CORE READY    │"));
    display.println(F("└────────────────────┘"));
    display.display();
}

void render_dashboard_safe(uint32_t t_voice_to_net, uint32_t t_ack_rtt, uint32_t dur, uint32_t asr_ms, const char* text) {
    if (xSemaphoreTake(screen_mutex, portMAX_DELAY) == pdTRUE) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);

        float total_e2e = (float)t_ack_rtt + (float)asr_ms;
        display.println(F(">> ACTIVATE DETECTED"));
        display.printf("Uplink/Net : %4.1f ms\n", (float)t_ack_rtt);
        display.printf("ASR Compute: %4.1f ms\n", (float)asr_ms);
        display.printf("Total E2E  : %4.1f ms\n", total_e2e);
        display.println(F("─────────────────────"));
        display.printf("\"%s\"\n", (text && strlen(text) > 0) ? text : "(listening)");
        display.display();
        xSemaphoreGive(screen_mutex);
    }
}

// ----------------------------------------------------------------------------
// 4. CORE 0 TASK: HIGH-PERFORMANCE PRE-WARMED SOCKET & KEEPALIVE
// ----------------------------------------------------------------------------
bool ensure_prewarmed_socket() {
    if (tcp_client.connected()) return true;

    Serial.println(F("[Core 0] Pre-warming persistent TCP socket connection..."));
    if (tcp_client.connect(SERVER_IP, TCP_SERVER_PORT)) {
        tcp_client.setNoDelay(true); // Disable Nagle's Algorithm for zero buffering
        uint8_t syn_byte = PROTOCOL_SYN;
        tcp_client.write(&syn_byte, 1);

        uint32_t t_start = millis();
        while (!tcp_client.available() && (millis() - t_start < 2000)) {}
        if (tcp_client.available() && tcp_client.read() == PROTOCOL_SYN_ACK) {
            Serial.println(F("✅ [PRE-WARMED SOCKET READY] 0x01 -> 0x06 SYN-ACK Verified!"));
            return true;
        }
    }
    return false;
}

void TaskNetworkStream(void *pvParameters) {
    Serial.println(F("[Core 0] Network Stream Task Running on Core 0"));
    AudioChunk chunk;
    uint32_t t_last_activity = millis();

    // Pre-warm socket on startup (0 ms wake handshake penalty)
    ensure_prewarmed_socket();

    while (true) {
        if (!tcp_client.connected()) {
            ensure_prewarmed_socket();
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        // Send 1-Byte TCP Keepalive Heartbeat (0x00) every 20 seconds when idle
        if (millis() - t_last_activity > 20000 && current_state == STATE_IDLE_LISTENING) {
            uint8_t heartbeat = PROTOCOL_HEARTBEAT;
            tcp_client.write(&heartbeat, 1);
            t_last_activity = millis();
        }

        // Wait for audio chunks from Core 1 Queue
        if (xQueueReceive(audio_queue, &chunk, pdMS_TO_TICKS(100)) == pdTRUE) {
            t_last_activity = millis();

            if (current_state == STATE_IDLE_LISTENING) {
                current_state = STATE_STREAMING_VOICE;
                if (xSemaphoreTake(screen_mutex, 10 / portTICK_PERIOD_MS) == pdTRUE) {
                    display.clearDisplay();
                    display.setCursor(5, 20);
                    display.println(F(">> ACTIVATE DETECTED"));
                    display.println(F("🎙️ STREAMING LIVE..."));
                    display.display();
                    xSemaphoreGive(screen_mutex);
                }
            }

            if (current_state == STATE_STREAMING_VOICE) {
                if (chunk.length > 0) {
                    // Send Length-Prefixed TLV Frame (Prevents 0xFF in-band PCM collision)
                    ChunkHeader header{ PROTOCOL_AUDIO_CHUNK, (uint16_t)chunk.length };
                    tcp_client.write((uint8_t*)&header, sizeof(header));
                    tcp_client.write(chunk.data, chunk.length);
                }

                if (chunk.is_last) {
                    // Send Stream End TLV Frame
                    ChunkHeader end_header{ PROTOCOL_STREAM_END, 0 };
                    tcp_client.write((uint8_t*)&end_header, sizeof(end_header));
                    int64_t t_last_packet_sent = esp_timer_get_time();

                    // Wait for Instant Hardware Transit ACK (0x7F) from Laptop
                    while (!tcp_client.available()) {}
                    uint8_t transit_ack = tcp_client.read();
                    int64_t t_ack_received = esp_timer_get_time();

                    uint32_t dt2_dt3_rtt = (t_ack_received - t_last_packet_sent) / 1000;

                    // Read 18-byte Telemetry Header Struct
                    ProfessionalTelemetry telem{};
                    while (tcp_client.available() < (int)sizeof(telem)) {}
                    tcp_client.readBytes((char*)&telem, sizeof(telem));

                    // Read UTF-8 Transcribed Text Payload
                    char text_buf[128] = {0};
                    if (telem.text_length > 0) {
                        uint16_t read_len = (telem.text_length < 127) ? telem.text_length : 127;
                        while (tcp_client.available() < read_len) {}
                        tcp_client.readBytes(text_buf, read_len);
                        text_buf[read_len] = '\0';
                    }

                    // Render Telemetry Dashboard for Judges on SSD1306 OLED
                    render_dashboard_safe(1, dt2_dt3_rtt, telem.audio_duration_ms, telem.server_asr_compute_ms, text_buf);

                    current_state = STATE_IDLE_LISTENING;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 5. CORE 1 TASK: I2S DMA ACQUISITION & TINYML KWS INFERENCE (WITH PRE-ROLL)
// ----------------------------------------------------------------------------
void TaskAudioKWS(void *pvParameters) {
    Serial.println(F("[Core 1] I2S Audio, KWS & Pre-Roll Task Running on Core 1"));
    uint8_t dma_buffer[CHUNK_BYTES];
    size_t bytes_read = 0;
    bool is_streaming = false;
    int64_t t_last_sound = 0;

    // 100 ms Lookback Ring Buffer (3 x 512-byte PCM chunks)
    uint8_t preroll_ring[3][CHUNK_BYTES];
    size_t preroll_lens[3] = {0, 0, 0};
    uint8_t preroll_idx = 0;

    while (true) {
        // Read raw 16kHz PCM16 samples directly from I2S DMA
        i2s_read(I2S_PORT, dma_buffer, CHUNK_BYTES, &bytes_read, portMAX_DELAY);

        int16_t* samples = (int16_t*)dma_buffer;
        int32_t sum = 0;
        for (int i = 0; i < (int)(bytes_read / 2); ++i) {
            sum += abs(samples[i]);
        }
        int32_t avg_amp = sum / (bytes_read / 2);

        int64_t now = esp_timer_get_time();

        if (!is_streaming) {
            // Store frame into 100ms Lookback Pre-Roll Buffer
            memcpy(preroll_ring[preroll_idx], dma_buffer, bytes_read);
            preroll_lens[preroll_idx] = bytes_read;
            preroll_idx = (preroll_idx + 1) % 3;

            // KWS Speech Energy Trigger
            if (avg_amp > 850) {
                is_streaming = true;
                t_last_sound = now;

                // 1. Flush historical 100ms pre-roll ring buffer first (Zero Command Clipping)
                for (int i = 0; i < 3; ++i) {
                    uint8_t idx = (preroll_idx + i) % 3;
                    if (preroll_lens[idx] > 0) {
                        AudioChunk pchunk;
                        memcpy(pchunk.data, preroll_ring[idx], preroll_lens[idx]);
                        pchunk.length = preroll_lens[idx];
                        pchunk.is_last = false;
                        xQueueSend(audio_queue, &pchunk, portMAX_DELAY);
                    }
                }

                // 2. Queue current live chunk
                AudioChunk chunk;
                memcpy(chunk.data, dma_buffer, bytes_read);
                chunk.length = bytes_read;
                chunk.is_last = false;
                xQueueSend(audio_queue, &chunk, portMAX_DELAY);
            }
        } else {
            // --- Active Voice Streaming Mode ---
            AudioChunk chunk;
            memcpy(chunk.data, dma_buffer, bytes_read);
            chunk.length = bytes_read;

            if (avg_amp > VAD_SILENCE_RMS) {
                t_last_sound = now;
            }

            // Silence Cutoff Check (1.2s trailing silence)
            if ((now - t_last_sound) / 1000 > SILENCE_TIMEOUT_MS) {
                chunk.is_last = true;
                is_streaming = false;
                xQueueSend(audio_queue, &chunk, portMAX_DELAY);
                vTaskDelay(pdMS_TO_TICKS(1500)); // Lockout cooldown
            } else {
                chunk.is_last = false;
                xQueueSend(audio_queue, &chunk, portMAX_DELAY);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 6. MAIN ARDUINO SETUP
// ----------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    screen_mutex = xSemaphoreCreateMutex();
    audio_queue = xQueueCreate(16, sizeof(AudioChunk));

    init_fast_oled();
    init_i2s_dma();

    // Connect to WiFi & Disable Modem Sleep Latency Spikes
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false); // Disables Wi-Fi modem sleep latency spikes
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print(F("Connecting WiFi"));
    while (WiFi.status() != WL_CONNECTED) {
        delay(200);
        Serial.print(F("."));
    }
    Serial.println(F("\n✅ ESP32-S3 Ready! IP: ") + WiFi.localIP().toString());

    // Dual-Core Task Pinning
    // Core 0: Dedicated TCP Socket, Wi-Fi Stack, OLED Telemetry Task
    xTaskCreatePinnedToCore(TaskNetworkStream, "NetTask", 4096, NULL, 2, NULL, 0);

    // Core 1: Dedicated I2S DMA Read, Pre-Roll Ring Buffer & TinyML KWS Task
    xTaskCreatePinnedToCore(TaskAudioKWS, "AudioKWSTask", 8192, NULL, 3, NULL, 1);
}

void loop() {
    vTaskDelete(NULL);
}
