import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
N_PARAMS = 5   # <---- change this for different parameter counts

WEIGHTS_PATH = f"encoder_{N_PARAMS}params.weights.h5"
ENCODE_DIR = "airfoils_png"
OUT_CSV = f"airfoil_latent_params_{N_PARAMS}.csv"
OUT_NPY = f"airfoil_latent_params_{N_PARAMS}.npy"

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# LOAD PNG FOLDER
# -----------------------------
def load_png_folder(folder, img_size=IMG_SIZE):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort()
    if not files:
        raise RuntimeError(f"No PNGs found in {folder}")
    X = []
    for f in files:
        p = os.path.join(folder, f)
        img = load_img(p, target_size=(img_size, img_size), color_mode="grayscale")
        arr = img_to_array(img) / 255.0
        X.append(arr)
    return np.array(X, dtype="float32"), files

X_imgs, file_list = load_png_folder(ENCODE_DIR)
print("Loaded PNGs. Shape:", X_imgs.shape)

# -----------------------------
# BUILD CNN ENCODER
# -----------------------------
inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

x = Conv2D(32, (3,3), activation='relu', padding='valid')(inp)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((3,3))(x)
x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((3,3))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.05)(x)

param_layer_name = f"p{N_PARAMS}"
params_pred = Dense(N_PARAMS, activation='linear', name=param_layer_name)(x)

encoder = Model(inp, params_pred, name=f"encoder_{N_PARAMS}")
encoder.compile(optimizer=Adam(5e-4), loss="mse")
encoder.summary()

# -----------------------------
# TRAIN IF WEIGHTS DO NOT EXIST
# -----------------------------
if os.path.exists(WEIGHTS_PATH):
    encoder.load_weights(WEIGHTS_PATH)
    print("Loaded encoder weights from", WEIGHTS_PATH)
else:
    print(f"No weights found for {N_PARAMS}-parameter model. Training from scratch...")

    # Dummy training targets (for unsupervised pretraining)
    params = np.zeros((len(file_list), N_PARAMS), dtype=np.float32)

    cbs = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    history = encoder.fit(
        X_imgs,
        params,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    # Save model weights
    encoder.save_weights(WEIGHTS_PATH)
    print(f"Saved encoder weights to {WEIGHTS_PATH}")

    # Evaluate final loss on the full dataset
    params_dummy = np.zeros((len(file_list), N_PARAMS), dtype=np.float32)
    full_loss = encoder.evaluate(X_imgs, params_dummy, batch_size=BATCH_SIZE, verbose=1)
    print("Loss on full dataset:", full_loss)

    # -----------------------------
    # Log the loss to a CSV file
    # -----------------------------
    log_csv = "training_log.csv"
    best_val_loss = min(history.history["val_loss"])
    file_exists = os.path.exists(log_csv)

    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["N_PARAMS", "final_loss", "best_val_loss"])
        writer.writerow([N_PARAMS, full_loss, best_val_loss])

    print(f"Logged results to {log_csv}")

# -----------------------------
# ENCODE PNGs → PARAMS
# -----------------------------
params_out = encoder.predict(X_imgs, batch_size=BATCH_SIZE, verbose=1)
print("Encoded params shape:", params_out.shape)

# Save CSV and NPY
np.save(OUT_NPY, params_out)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    header = ["filename"] + [f"p{i+1}" for i in range(N_PARAMS)]
    w.writerow(header)
    for fname, vec in zip(file_list, params_out):
        w.writerow([fname] + list(map(float, vec)))

print(f"Saved: {OUT_CSV} and {OUT_NPY}")
