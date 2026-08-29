"""
Ultra-lightweight Single Keyword Spotting (KWS) training + INT8 quantization pipeline
for edge/TinyML deployment (ESP32 / Cortex-M class devices).

Target Keyword: "activate" (3 Classes: background, unknown, activate)
"""

import os
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Configuration (Single Keyword: activate)
# ---------------------------------------------------------------------------
DATASET_PATH = "vikram_split_dataset"
CLASSES = ["background", "unknown", "activate"]
NUM_CLASSES = len(CLASSES)
KEYWORD_INDICES = {CLASSES.index("activate")}

SAMPLE_RATE = 16000
DURATION = 1.00
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)          # 16000 samples

# Standard 40ms window / 20ms hop -> 51 raw frames for a 1s clip, center-trimmed to 49
N_FFT = 640          # 40ms @ 16kHz
HOP_LENGTH = 320      # 20ms @ 16kHz
N_MELS = 40
N_FRAMES = 49
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)

AUG_MULTIPLIER = {"activate": 4, "unknown": 1, "background": 1}
NOISE_PROB = 0.6       # Probability an augmented sample gets noise mixed in
NOISE_SNR_DB_RANGE = (3, 15)


# ---------------------------------------------------------------------------
# 2. Feature extraction (Produces 40x49 log-mel matrix)
# ---------------------------------------------------------------------------
def extract_res8_features(audio: np.ndarray) -> np.ndarray:
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    n = log_mel.shape[1]
    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        mat = log_mel[:, start:start + N_FRAMES]
    else:
        mat = np.pad(log_mel, ((0, 0), (0, N_FRAMES - n)), mode="constant")

    # Min-Max normalize to [0, 1] for INT8 quantization stability
    return np.clip((mat + 80.0) / 80.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Augmentation helpers
# ---------------------------------------------------------------------------
def load_noise_clips(train_paths, train_labels):
    bg_idx = CLASSES.index("background")
    train_bg_files = [p for p, l in zip(train_paths, train_labels) if l == bg_idx]
    if not train_bg_files:
        print("[WARN] No background files in training split; skipping noise augmentation.")
        return []
    clips = []
    for path in train_bg_files:
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        if len(y) >= TARGET_SAMPLES:
            clips.append(y)
    return clips


def mix_noise(audio: np.ndarray, noise_clips: list, snr_db_range=NOISE_SNR_DB_RANGE) -> np.ndarray:
    if not noise_clips:
        return audio
    noise = random.choice(noise_clips)
    start = random.randint(0, max(0, len(noise) - TARGET_SAMPLES))
    noise_seg = noise[start:start + TARGET_SAMPLES]
    if len(noise_seg) < TARGET_SAMPLES:
        noise_seg = np.pad(noise_seg, (0, TARGET_SAMPLES - len(noise_seg)))

    sig_power = np.mean(audio ** 2) + 1e-9
    noise_power = np.mean(noise_seg ** 2) + 1e-9
    snr_db = random.uniform(*snr_db_range)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise_seg = noise_seg * np.sqrt(target_noise_power / noise_power)
    return audio + noise_seg


def time_shift(audio: np.ndarray, max_shift_samples: int = 3200) -> np.ndarray:
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_SAMPLES]

    shift = random.randint(-max_shift_samples, max_shift_samples)
    if shift > 0:
        aud = np.pad(audio, (shift, 0), mode="constant")[:TARGET_SAMPLES]
    elif shift < 0:
        aud = np.pad(audio, (0, -shift), mode="constant")[-shift:-shift + TARGET_SAMPLES]
    else:
        aud = audio
    return aud


def augment(audio: np.ndarray, noise_clips: list) -> np.ndarray:
    aud = time_shift(audio)
    aud = aud * random.uniform(0.80, 1.20)
    if random.random() < NOISE_PROB:
        aud = mix_noise(aud, noise_clips)
    return aud


# ---------------------------------------------------------------------------
# 4. Depthwise-separable residual block
# ---------------------------------------------------------------------------
def ds_res_block(input_tensor, out_channels, stride=1):
    x = layers.DepthwiseConv2D((3, 3), strides=stride, padding="same", use_bias=False)(input_tensor)
    x = layers.Conv2D(out_channels, (1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=1, padding="same", use_bias=False)(x)
    x = layers.Conv2D(out_channels, (1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    shortcut = input_tensor
    if stride != 1 or input_tensor.shape[-1] != out_channels:
        shortcut = layers.Conv2D(out_channels, (1, 1), strides=stride, padding="same", use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


# ---------------------------------------------------------------------------
# 5. Data pipeline
# ---------------------------------------------------------------------------
def get_loudest_window(audio: np.ndarray, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    if len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)), mode="constant")
    elif len(audio) == target_samples:
        return audio

    max_energy = -1
    best_start = 0
    for start in range(0, len(audio) - target_samples + 1, 800):
        window = audio[start:start + target_samples]
        energy = np.sum(window ** 2)
        if energy > max_energy:
            max_energy = energy
            best_start = start
    return audio[best_start:best_start + target_samples]


def build_split(paths, labels, augment_multiplier, apply_augmentation, noise_clips):
    X, Y = [], []
    for path, label in zip(paths, labels):
        c_name = CLASSES[label]
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        audio = get_loudest_window(audio)

        # Include clean baseline
        X.append(extract_res8_features(audio))
        Y.append(label)

        if apply_augmentation:
            reps = augment_multiplier.get(c_name, 1) - 1
            for _ in range(max(0, reps)):
                aug_audio = augment(audio, noise_clips)
                X.append(extract_res8_features(aug_audio))
                Y.append(label)
    return np.array(X)[..., np.newaxis], np.array(Y)


if __name__ == "__main__":
    print("[DATA] Indexing source files for 3-class model ('activate')...")
    file_paths, file_labels = [], []
    for idx, c_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, c_name)
        if not os.path.exists(folder):
            print(f"[WARN] Folder missing: {folder}")
            continue
        for f in os.listdir(folder):
            if f.endswith(".wav"):
                file_paths.append(os.path.join(folder, f))
                file_labels.append(idx)

    if not file_paths:
        raise RuntimeError(f"No .wav files found under {DATASET_PATH}. Check CLASSES/paths.")

    # Train/Validation split at file level
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, file_labels, test_size=0.15, random_state=SEED, stratify=file_labels
    )

    noise_clips = load_noise_clips(train_files, train_labels)

    print("[DATA] Building training set (augmented)...")
    X_train, Y_train = build_split(
        train_files, train_labels, AUG_MULTIPLIER,
        apply_augmentation=True, noise_clips=noise_clips,
    )

    print("[DATA] Building validation set (clean)...")
    X_val, Y_val = build_split(
        val_files, val_labels, AUG_MULTIPLIER,
        apply_augmentation=False, noise_clips=[],
    )

    print(f"[DATA] X_train={X_train.shape}, X_val={X_val.shape}")
    for i, name in enumerate(CLASSES):
        n_tr = int(np.sum(Y_train == i))
        n_va = int(np.sum(Y_val == i))
        print(f"  {name:12s} train={n_tr:5d}  val={n_va:4d}")

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_CLASSES), y=Y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"[DATA] class_weight={class_weight}")

    # -----------------------------------------------------------------
    # 6. Build model
    # -----------------------------------------------------------------
    inputs = layers.Input(shape=INPUT_SHAPE)

    x = layers.Conv2D(16, (3, 3), strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = ds_res_block(x, out_channels=16, stride=1)
    x = ds_res_block(x, out_channels=24, stride=2)
    x = ds_res_block(x, out_channels=32, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # -----------------------------------------------------------------
    # 7. Train
    # -----------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]

    print("\nTraining depthwise-separable Res8 single-keyword model...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=40,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save("res8_activate_float.keras")
    print("[TRAIN] Saved float Keras model 'res8_activate_float.keras'")

    # -----------------------------------------------------------------
    # 8. Evaluate: Confusion matrix + FAR / FRR
    # -----------------------------------------------------------------
    val_probs = model.predict(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    print("\n[EVAL] Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(Y_val, val_preds))

    print("\n[EVAL] Per-class precision/recall:")
    print(classification_report(Y_val, val_preds, target_names=CLASSES, digits=3))

    non_keyword_mask = ~np.isin(Y_val, list(KEYWORD_INDICES))
    false_accepts = np.isin(val_preds[non_keyword_mask], list(KEYWORD_INDICES)).sum()
    far = false_accepts / max(1, non_keyword_mask.sum())
    print(f"\n[EVAL] False-accept rate (non-keyword -> keyword): {far:.4%} "
          f"({false_accepts}/{non_keyword_mask.sum()})")

    keyword_mask = np.isin(Y_val, list(KEYWORD_INDICES))
    false_rejects = (~np.isin(val_preds[keyword_mask], list(KEYWORD_INDICES))).sum()
    frr = false_rejects / max(1, keyword_mask.sum())
    print(f"[EVAL] False-reject rate (keyword -> non-keyword): {frr:.4%} "
          f"({false_rejects}/{keyword_mask.sum()})")

    # -----------------------------------------------------------------
    # 9. Full INT8 quantization
    # -----------------------------------------------------------------
    print("\nQuantizing to full INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_dataset_gen():
        n = min(150, len(X_train))
        idxs = np.random.choice(len(X_train), n, replace=False)
        for i in idxs:
            yield [X_train[i][np.newaxis, ...].astype(np.float32)]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()

    out_path = "res8_activate_int8.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nSaved '{out_path}' ({size_kb:.1f} KB flash for weights).")

    # Emit C byte array for embedded firmware
    c_array_path = "res8_activate_int8.cc"
    with open(c_array_path, "w") as f:
        f.write("// Auto-generated from res8_activate_int8.tflite\n")
        f.write("#include <cstdint>\n\n")
        f.write("alignas(8) const unsigned char res8_activate_int8_tflite[] = {\n")
        for i in range(0, len(tflite_model), 12):
            chunk = tflite_model[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int res8_activate_int8_tflite_len = {len(tflite_model)};\n")
    print(f"Saved '{c_array_path}' for direct inclusion in firmware.")