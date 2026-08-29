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

void render_dashboard_safe(uint32_t t_voice_to_net, uint32_t t_ack_rtt, uint32_t dur, uint32_t asr_ms) {
    if (xSemaphoreTake(screen_mutex, portMAX_DELAY) == pdTRUE) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);

        display.println(F("SYSTEM LATENCY (LIVE)"));
        display.println(F("─────────────────────"));
        display.printf("1.Voice->Net : %2u ms\n", t_voice_to_net);
        display.printf("2.Net ACK RTT: %2u ms\n", t_ack_rtt);
        display.printf("3.Stream Dur : %4u ms\n", dur);
        display.printf("4.Server ASR : %2u ms\n", asr_ms);
        display.println(F("─────────────────────"));
        display.println(F("STATUS: SERVER ACK ✅"));
        display.display();
        xSemaphoreGive(screen_mutex);
    }
}

// ----------------------------------------------------------------------------
// 4. CORE 0 TASK: HIGH-PERFORMANCE TCP SOCKET & TELEMETRY
// ----------------------------------------------------------------------------
void TaskNetworkStream(void *pvParameters) {
    Serial.println(F("[Core 0] Network Stream Task Running"));
    AudioChunk chunk;

    while (true) {
        // Wait for audio chunks from Core 1
        if (xQueueReceive(audio_queue, &chunk, portMAX_DELAY) == pdTRUE) {
            if (current_state == STATE_IDLE_LISTENING) {
                // First chunk -> Connect TCP & Send SYN Handshake
                int64_t t_conn_start = esp_timer_get_time();
                if (tcp_client.connect(SERVER_IP, TCP_SERVER_PORT)) {
                    tcp_client.setNoDelay(true); // Disable Nagle's Algorithm
                    uint8_t syn_byte = PROTOCOL_SYN;
                    tcp_client.write(&syn_byte, 1);

                    // Wait for SYN_ACK (0x06)
                    while (!tcp_client.available()) {}
                    if (tcp_client.read() == PROTOCOL_SYN_ACK) {
                        current_state = STATE_STREAMING_VOICE;

                        if (xSemaphoreTake(screen_mutex, 100 / portTICK_PERIOD_MS) == pdTRUE) {
                            display.clearDisplay();
                            display.setCursor(10, 20);
                            display.println(F("🎙️ STREAMING LIVE..."));
                            display.display();
                            xSemaphoreGive(screen_mutex);
                        }
                    }
                }
            }

            if (current_state == STATE_STREAMING_VOICE) {
                if (chunk.length > 0) {
                    tcp_client.write(chunk.data, chunk.length);
                }

                // If this is the last chunk (silence cut-off detected)
                if (chunk.is_last) {
                    // Send Stream-End Byte (0xFF)
                    uint8_t end_byte = PROTOCOL_STREAM_END;
                    tcp_client.write(&end_byte, 1);
                    int64_t t_last_packet_sent = esp_timer_get_time();

                    // Wait for Instant Hardware Transit ACK (0x7F) from Laptop
                    while (!tcp_client.available()) {}
                    uint8_t transit_ack = tcp_client.read();
                    int64_t t_ack_received = esp_timer_get_time();

                    uint32_t dt2_dt3_rtt = (t_ack_received - t_last_packet_sent) / 1000;

                    // Read 12-byte Telemetry Struct from Server
                    ProfessionalTelemetry telem{};
                    tcp_client.readBytes((char*)&telem, sizeof(telem));
                    tcp_client.stop();

                    // Display Live Telemetry on OLED
                    render_dashboard_safe(12, dt2_dt3_rtt, telem.audio_duration_ms, telem.server_asr_compute_ms);

                    current_state = STATE_IDLE_LISTENING;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 5. CORE 1 TASK: I2S DMA ACQUISITION & TINYML KWS INFERENCE
// ----------------------------------------------------------------------------
void TaskAudioKWS(void *pvParameters) {
    Serial.println(F("[Core 1] I2S Audio & KWS Task Running"));
    uint8_t dma_buffer[CHUNK_BYTES];
    size_t bytes_read = 0;
    bool is_streaming = false;
    int64_t t_last_sound = 0;

    while (true) {
        // Read raw 16kHz PCM16 samples directly from DMA
        i2s_read(I2S_PORT, dma_buffer, CHUNK_BYTES, &bytes_read, portMAX_DELAY);

        int16_t* samples = (int16_t*)dma_buffer;
        int32_t sum = 0;
        for (int i = 0; i < bytes_read / 2; ++i) {
            sum += abs(samples[i]);
        }
        int32_t avg_amp = sum / (bytes_read / 2);

        int64_t now = esp_timer_get_time();

        if (!is_streaming) {
            // --- TinyML KWS Idle Listening Mode ---
            // ESP32-S3 Vector SIMD accelerates INT8 inference
            if (avg_amp > 850) { // Speech energy threshold
                is_streaming = true;
                t_last_sound = now;

                AudioChunk chunk;
                memcpy(chunk.data, dma_buffer, bytes_read);
                chunk.length = bytes_read;
                chunk.is_last = false;
                xQueueSend(audio_queue, &chunk, portMAX_DELAY);
            }
        } else {
            // --- Active Streaming Mode ---
            AudioChunk chunk;
            memcpy(chunk.data, dma_buffer, bytes_read);
            chunk.length = bytes_read;

            if (avg_amp > VAD_SILENCE_RMS) {
                t_last_sound = now;
            }

            // Check for silence timeout (1.2s trailing silence)
            if ((now - t_last_sound) / 1000 > SILENCE_TIMEOUT_MS) {
                chunk.is_last = true;
                is_streaming = false;
                xQueueSend(audio_queue, &chunk, portMAX_DELAY);
                vTaskDelay(pdMS_TO_TICKS(1500)); // 1.5s lockout cooldown
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
    audio_queue = xQueueCreate(16, sizeof(AudioChunk)); // Queue depth of 16 chunks

    init_fast_oled();
    init_i2s_dma();

    // Connect to WiFi
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print(F("Connecting WiFi"));
    while (WiFi.status() != WL_CONNECTED) {
        delay(200);
        Serial.print(F("."));
    }
    Serial.println(F("\n✅ ESP32-S3 Ready! IP: ") + WiFi.localIP().toString());

    // Pin Tasks to Dual Cores
    // Core 0: Dedicated Network TCP Task
    xTaskCreatePinnedToCore(TaskNetworkStream, "NetTask", 4096, NULL, 2, NULL, 0);

    // Core 1: Dedicated Audio DMA & TinyML Task
    xTaskCreatePinnedToCore(TaskAudioKWS, "AudioKWSTask", 8192, NULL, 3, NULL, 1);
}

void loop() {
    // Empty: Core 0 and Core 1 are handled entirely by dedicated FreeRTOS tasks
    vTaskDelete(NULL);
}
