import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Configuration (Matches single-keyword 'activate' model)
# ---------------------------------------------------------------------------
MODEL_PATH = "res8_activate_int8.tflite"
CLASSES = ["background", "unknown", "activate"]

SAMPLE_RATE = 16000
DURATION = 1.00
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)  # 16000 samples

N_FFT = 640        # 40ms window @ 16kHz
HOP_LENGTH = 320   # 20ms hop @ 16kHz
N_MELS = 40
N_FRAMES = 49      # Center-trimmed to 49 frames

# Class Index
ACTIVATE_IDX = CLASSES.index("activate")

# Detection Thresholds & Timing
ACTIVATE_THRESHOLD = 0.65     # Trigger confidence threshold
COOLDOWN_SEC = 1.5            # Lockout period after activation
VAD_RMS_THRESHOLD = 0.005     # Silence gate threshold

# ---------------------------------------------------------------------------
# Load and Initialize TFLite INT8 Engine
# ---------------------------------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_idx = input_details['index']
out_idx = output_details['index']
in_scale, in_zero_point = input_details['quantization']
out_scale, out_zero_point = output_details['quantization']

# Rolling audio buffer
audio_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)

# ---------------------------------------------------------------------------
# Feature Extraction (Mirrors training preprocessing)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Audio Callback Stream Handler
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# ---------------------------------------------------------------------------
# Real-Time UI Setup (Matplotlib GUI)
# ---------------------------------------------------------------------------
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.3, 1.2])

# Subplot 1: Raw Audio Waveform
ax_wave = fig.add_subplot(gs[0, :])
line_wave, = ax_wave.plot(np.linspace(0, 1, TARGET_SAMPLES), np.zeros(TARGET_SAMPLES), color="#00ffcc", lw=1)
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.set_xlim(0, 1)
ax_wave.set_title("Live Audio Input (1.0s Rolling Buffer)", fontsize=11, fontweight="bold")
ax_wave.set_ylabel("Amplitude")
ax_wave.grid(True, linestyle="--", alpha=0.3)

# Subplot 2: Log-Mel Spectrogram Matrix (40x49)
ax_spec = fig.add_subplot(gs[1, 0])
im_spec = ax_spec.imshow(np.zeros((N_MELS, N_FRAMES)), origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=1)
ax_spec.set_title(f"Model Input Features ({N_MELS} Mels x {N_FRAMES} Frames)", fontsize=11, fontweight="bold")
ax_spec.set_xlabel("Time Frames")
ax_spec.set_ylabel("Mel Bins")

# Subplot 3: Real-Time Confidence Bar Chart (3 Classes)
ax_bars = fig.add_subplot(gs[1, 1])
bar_colors = ["#7f8c8d", "#95a5a6", "#2ecc71"]
bars = ax_bars.bar(CLASSES, [0.0, 0.0, 0.0], color=bar_colors)
ax_bars.set_ylim(0, 1.0)
ax_bars.set_title("Current Class Probabilities", fontsize=11, fontweight="bold")
ax_bars.set_ylabel("Confidence")
ax_bars.grid(axis='y', linestyle="--", alpha=0.3)

# Subplot 4: Rolling Confidence History ('activate')
ax_history = fig.add_subplot(gs[2, :])
HISTORY_LEN = 100
hist_act = np.zeros(HISTORY_LEN)
hist_unk = np.zeros(HISTORY_LEN)

line_act, = ax_history.plot(hist_act, label="Activate Conf", color="#2ecc71", lw=2)
line_unk, = ax_history.plot(hist_unk, label="Unknown Conf", color="#95a5a6", lw=1, linestyle=":")
ax_history.axhline(y=ACTIVATE_THRESHOLD, color='#2ecc71', linestyle='--', alpha=0.8, label=f'Threshold ({ACTIVATE_THRESHOLD})')
ax_history.set_ylim(0, 1.05)
ax_history.set_xlim(0, HISTORY_LEN)
ax_history.set_title("Keyword Detection Probability Stream", fontsize=11, fontweight="bold")
ax_history.set_ylabel("Probability")
ax_history.legend(loc="upper left", ncol=3, fontsize=9)
ax_history.grid(True, linestyle="--", alpha=0.3)

status_text = ax_history.text(0.5, 0.85, "STATE: LISTENING", transform=ax_history.transAxes,
                              ha='center', va='center', fontsize=12, fontweight='bold',
                              color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', alpha=0.8))

plt.tight_layout()
plt.ion()
plt.show()

# ---------------------------------------------------------------------------
# Execution & Visualization Loop
# ---------------------------------------------------------------------------
step_size = int(SAMPLE_RATE * 0.06)  # 60ms streaming step
last_trigger_time = 0

try:
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=step_size, callback=audio_callback):
        while plt.fignum_exists(fig.number):
            current_time = time.time()

            # 1. Post-Trigger Cooldown Handler
            if current_time - last_trigger_time < COOLDOWN_SEC:
                audio_buffer.fill(0)
                probs = np.zeros(len(CLASSES))
                status_text.set_text("STATE: COOLDOWN")
                status_text.get_bbox_patch().set_facecolor('#d35400')
                feat = np.zeros((N_MELS, N_FRAMES))
            else:
                # 2. VAD / Silence Check
                rms = np.sqrt(np.mean(audio_buffer**2))
                if rms < VAD_RMS_THRESHOLD:
                    probs = np.zeros(len(CLASSES))
                    probs[0] = 1.0  # Background
                    feat = extract_features(audio_buffer)
                    status_text.set_text("STATE: LISTENING (Silent)")
                    status_text.get_bbox_patch().set_facecolor('#2c3e50')
                else:
                    # 3. Feature Extraction & Model Inference
                    feat = extract_features(audio_buffer)
                    inp = feat[np.newaxis, ..., np.newaxis].astype(np.float32)

                    if in_scale > 0:
                        inp = np.round(inp / in_scale + in_zero_point).astype(np.int8)

                    interpreter.set_tensor(in_idx, inp)
                    interpreter.invoke()
                    out = interpreter.get_tensor(out_idx)

                    if out_scale > 0:
                        probs = (out.astype(np.float32) - out_zero_point) * out_scale
                    else:
                        probs = out
                    probs = probs[0]

                    act_p = probs[ACTIVATE_IDX]

                    # 4. Trigger Decision Logic
                    if act_p >= ACTIVATE_THRESHOLD:
                        status_text.set_text(f"🔥 TRIGGER MATCHED: 'ACTIVATE' ({act_p*100:.1f}%) 🔥")
                        status_text.get_bbox_patch().set_facecolor('#27ae60')
                        print(f"\n🎯 [TRIGGER] Keyword 'Activate' detected ({act_p*100:.1f}%)!")
                        last_trigger_time = current_time
                        audio_buffer.fill(0)
                    else:
                        status_text.set_text("STATE: LISTENING")
                        status_text.get_bbox_patch().set_facecolor('#2c3e50')

            # 5. Fast GUI Update
            line_wave.set_ydata(audio_buffer)
            im_spec.set_data(feat)

            for bar, p in zip(bars, probs):
                bar.set_height(p)

            hist_act = np.roll(hist_act, -1)
            hist_unk = np.roll(hist_unk, -1)
            hist_act[-1] = probs[ACTIVATE_IDX]
            hist_unk[-1] = probs[CLASSES.index("unknown")]

            line_act.set_ydata(hist_act)
            line_unk.set_ydata(hist_unk)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] Streaming visualization stopped cleanly.")
finally:
    plt.close('all')