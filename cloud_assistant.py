#!/usr/bin/env python3
"""
=============================================================================
LOCAL MULTILINGUAL ASSISTANT (100% OFFLINE - FASTER-WHISPER)
• Isolates Deepgram cloud dependency -> Uses local Faster-Whisper (INT8 CPU)
• Supports Real-Time Multilingual STT: Hindi (हिंदी) & English (Hinglish)
• Local Wake-Word Engine ("Activate" TFLite INT8 model)
=============================================================================
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from faster_whisper import WhisperModel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration (Matches res8_activate_int8.tflite)
# ---------------------------------------------------------------------------
MODEL_PATH = "res8_activate_int8.tflite"
CLASSES = ["background", "unknown", "activate"]
ACTIVATE_IDX = CLASSES.index("activate")

SAMPLE_RATE = 16000
DURATION = 1.00
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 640
HOP_LENGTH = 320
N_MELS = 40
N_FRAMES = 49

ACTIVATE_THRESHOLD = 0.65
VAD_RMS_THRESHOLD = 0.005

# ---------------------------------------------------------------------------
# 1. Load Local Wake-Word INT8 TFLite Model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("🚀 [1/2] Loading Local Wake-Word Model ('Activate')...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']
in_scale, in_zero_point = interpreter.get_input_details()[0]['quantization']
out_scale, out_zero_point = interpreter.get_output_details()[0]['quantization']

audio_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)

# ---------------------------------------------------------------------------
# 2. Load Local Faster-Whisper ASR Engine (INT8 Multilingual: Hindi & English)
# ---------------------------------------------------------------------------
print("🚀 [2/2] Loading Offline Faster-Whisper ASR Engine (Hindi + English)...")
# 'base' or 'small' multilingual model for fast INT8 CPU inference
WHISPER_MODEL_SIZE = "base"
whisper_engine = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
print(f"✅ Offline ASR Engine Ready: Faster-Whisper '{WHISPER_MODEL_SIZE}' (INT8 Quantized)")
print("=" * 60)

def extract_features(audio: np.ndarray) -> np.ndarray:
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    n = log_mel.shape[1]
    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        mat = log_mel[:, start:start + N_FRAMES]
    else:
        mat = np.pad(log_mel, ((0, 0), (0, N_FRAMES - n)), mode="constant")

    return np.clip((mat + 80.0) / 80.0, 0.0, 1.0)

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# ---------------------------------------------------------------------------
# 3. Offline STT Processing (Hindi & English Support)
# ---------------------------------------------------------------------------
def run_offline_asr(silence_limit_sec=1.5, max_speech_sec=8.0):
    print("\n" + "—" * 60)
    print("🎙️ [OFFLINE RECORDING ACTIVE] Bolna shuru kijiye (Hindi/English)...")
    print("—" * 60)

    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    recorded_chunks = []
    silent_chunks = 0
    max_silent_chunks = int(silence_limit_sec / 0.1)
    start_time = time.time()

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16', blocksize=chunk_size) as mic:
        while True:
            data, _ = mic.read(chunk_size)
            recorded_chunks.append(data.copy())

            # Energy check for silence exit
            rms = np.sqrt(np.mean((data.astype(np.float32) / 32768.0) ** 2))
            if rms < 0.008:
                silent_chunks += 1
            else:
                silent_chunks = 0

            elapsed = time.time() - start_time
            if (silent_chunks >= max_silent_chunks and elapsed > 1.0) or (elapsed > max_speech_sec):
                print("\n🛑 [RECORDING FINISHED] Processing speech locally...")
                break

    # Convert recorded audio chunks to normalized float32 buffer
    raw_bytes = b"".join([c.tobytes() for c in recorded_chunks])
    if len(raw_bytes) == 0:
        return "", "unknown", 0.0

    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_dur_sec = len(audio_np) / SAMPLE_RATE

    # Transcribe with Faster-Whisper (auto-detect Hindi / English)
    t_asr_start = time.perf_counter()
    segments, info = whisper_engine.transcribe(
        audio_np,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    transcription = " ".join([seg.text for seg in segments]).strip()
    t_asr_end = time.perf_counter()

    asr_latency_ms = (t_asr_end - t_asr_start) * 1000
    detected_lang = info.language
    lang_probability = info.language_probability * 100

    return transcription, detected_lang, lang_probability, asr_latency_ms, audio_dur_sec

# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main():
    step_size = int(SAMPLE_RATE * 0.06)
    print("\n" + "=" * 60)
    print("🟢 SYSTEM READY: Local Wake-Word & Faster-Whisper STT Active!")
    print("   Say 'Activate' to trigger speech recognition.")
    print("=" * 60)

    try:
        while True:
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=step_size, callback=audio_callback):
                while True:
                    rms = np.sqrt(np.mean(audio_buffer ** 2))
                    if rms < VAD_RMS_THRESHOLD:
                        time.sleep(0.03)
                        continue

                    feat = extract_features(audio_buffer)
                    inp = feat[np.newaxis, ..., np.newaxis].astype(np.float32)

                    if in_scale > 0:
                        inp = np.round(inp / in_scale + in_zero_point).astype(np.int8)

                    interpreter.set_tensor(in_idx, inp)
                    interpreter.invoke()
                    out = interpreter.get_tensor(out_idx)

                    probs = (out.astype(np.float32) - out_zero_point) * out_scale if out_scale > 0 else out
                    act_prob = probs[0][ACTIVATE_IDX]
                    bar = "█" * int(act_prob * 15)
                    print(f"\r[LOCAL LISTEN] 'Activate': [{bar:<15}] {act_prob * 100:4.1f}%", end="")

                    if act_prob >= ACTIVATE_THRESHOLD:
                        print(f"\n⚡ 'Activate' Triggered ({act_prob * 100:.1f}%)! Starting Local STT...")
                        break

                    time.sleep(0.02)

            # Process STT locally using Faster-Whisper
            text, lang, lang_prob, latency_ms, duration_sec = run_offline_asr()

            lang_label = "Hindi 🇮🇳" if lang == "hi" else ("English 🇬🇧" if lang == "en" else f"{lang.upper()}")

            print("\n" + "┌" + "─" * 58 + "┐")
            print("│              📊 LOCAL FASTER-WHISPER RESULTS              │")
            print("├" + "─" * 58 + "┤")
            print(f"│ • Detected Language : {lang_label:<34} ({lang_prob:.1f}%) │")
            print(f"│ • Audio Duration    : {duration_sec:<38.2f} sec │")
            print(f"│ • ASR Compute Time  : {latency_ms:<38.2f} ms  │")
            print("├" + "─" * 58 + "┤")
            print(f"│ • Spoken Text       : \"{text}\"")
            print("└" + "─" * 58 + "┘\n")

            # Reset buffer for next listening cycle
            audio_buffer.fill(0)
            time.sleep(1.0)
            print("=" * 60)
            print("🟢 Resuming Wake-Word Listening...")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

if __name__ == "__main__":
    main()