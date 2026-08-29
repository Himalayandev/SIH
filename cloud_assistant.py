import os
import sys
import time
import asyncio
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration (Matches your folder's res8_activate_int8.tflite)
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = "0756cb24de452681cd05b321d97b100248b0a327"  # <-- Apni Deepgram API key yahan dalein

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
# Load INT8 Model
# ---------------------------------------------------------------------------
print("\n[1/2] Loading Local Wake-Word Model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']
in_scale, in_zero_point = interpreter.get_input_details()[0]['quantization']
out_scale, out_zero_point = interpreter.get_output_details()[0]['quantization']

audio_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)

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
# Cloud WebSocket Streaming ASR Session
# ---------------------------------------------------------------------------
async def run_cloud_asr(silence_limit_sec=1.8, max_speech_sec=8.0):
    print("\n" + "—"*55)
    print("🌐 [CLOUD STREAM ACTIVE] Bolna shuru kijiye...")
    print("—"*55)

    deepgram = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

    transcription_result = []
    is_speaking = True

    async def on_message(result, **kwargs):
        if not hasattr(result, "channel"):
            return
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) > 0:
            print(f"\r⚡ Live Text: {sentence}", end="", flush=True)
            if result.is_final:
                transcription_result.append(sentence)

    # Use the SDK v7 async context manager
    async with deepgram.listen.v1.connect(
        model="nova-2",
        language="en-IN",
        smart_format=True,
        encoding="linear16",
        channels=1,
        sample_rate=SAMPLE_RATE,
        interim_results=True,
        endpointing=300
    ) as dg_connection:

        dg_connection.on(EventType.MESSAGE, on_message)

        # Start the background task to listen for messages
        listen_task = asyncio.create_task(dg_connection.start_listening())

        chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms streaming chunks
        silent_chunks = 0
        max_silent_chunks = int(silence_limit_sec / 0.1)
        start_time = time.time()

        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16', blocksize=chunk_size) as mic:
            while is_speaking:
                data, _ = mic.read(chunk_size)
                await dg_connection.send_media(data.tobytes())

                # Energy check for silence exit
                rms = np.sqrt(np.mean((data.astype(np.float32) / 32768.0)**2))
                if rms < 0.008:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if (silent_chunks >= max_silent_chunks and (time.time() - start_time) > 1.2) or (time.time() - start_time > max_speech_sec):
                    print("\n🛑 [STOP] Silence detected, closing stream...")
                    is_speaking = False

                await asyncio.sleep(0.01)

        # Finalize the WebSocket connection and close nicely
        await dg_connection.send_finalize()
        await asyncio.sleep(0.5)

        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass

    return " ".join(transcription_result).strip()

# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main():
    step_size = int(SAMPLE_RATE * 0.06)
    print("\n" + "="*55)
    print("🟢 SYSTEM READY: Listening for 'Activate'...")
    print("="*55)

    try:
        while True:
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=step_size, callback=audio_callback):
                while True:
                    rms = np.sqrt(np.mean(audio_buffer**2))
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
                    print(f"\r[LOCAL LISTEN] 'Activate': [{bar:<15}] {act_prob*100:4.1f}%", end="")

                    if act_prob >= ACTIVATE_THRESHOLD:
                        print(f"\n⚡ 'Activate' Triggered ({act_prob*100:.1f}%)! Connecting to Cloud...")
                        break

                    time.sleep(0.02)

            # ASR Stream start
            command = asyncio.run(run_cloud_asr())
            print(f"\n✅ COMMAND RECEIVED: \"{command}\"\n")

            # Reset buffer for next listening cycle
            audio_buffer.fill(0)
            time.sleep(1.0)
            print("="*55)
            print("🟢 Resuming Listening...")
            print("="*55)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

if __name__ == "__main__":
    main()