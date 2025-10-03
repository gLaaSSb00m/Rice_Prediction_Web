import os, warnings, traceback
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2

# -----------------------------
# Strategy (GPU/CPU)
# -----------------------------
def get_strategy():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        strat = tf.distribute.MirroredStrategy()
        print(f"✅ Using MirroredStrategy on {len(gpus)} GPU(s).")
        return strat
    print("✅ Using default strategy (CPU).")
    return tf.distribute.get_strategy()

strategy = get_strategy()
print("Replicas:", strategy.num_replicas_in_sync)

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_PATH = os.path.join("models", "best_VGG16_stage2.weights.h5")
IMAGE_SIZE = (224, 224)

RICE_CLASSES = [
    "10_Lal_Aush","11_Jirashail","12_Gutisharna","13_Red_Cargo","14_Najirshail",
    "15_Katari_Polao","16_Lal_Biroi","17_Chinigura_Polao","18_Amondhan","19_Shorna5",
    "1_Subol_Lota","20_Lal_Binni","21_Arborio","22_Turkish_Basmati","23_Ipsala",
    "24_Jasmine","25_Karacadag","26_BD30","27_BD33","28_BD39","29_BD49",
    "2_Bashmoti","30_BD51","31_BD52","32_BD56","33_BD57","34_BD70","35_BD72",
    "36_BD75","37_BD76","38_BD79","39_BD85","3_Ganjiya","40_BD87","41_BD91",
    "42_BD93","43_BD95","44_Binadhan7","45_Binadhan8","46_Binadhan10","47_Binadhan11",
    "48_Binadhan12","49_Binadhan14","4_Shampakatari","50_Binadhan16","51_Binadhan17",
    "52_Binadhan19","53_Binadhan21","54_Binadhan23","55_Binadhan24","56_Binadhan25",
    "57_Binadhan26","58_BR22","59_BR23","5_Katarivog","60_BRRI67","61_BRRI74",
    "62_BRRI102","6_BR28","7_BR29","8_Paijam","9_Bashful"
]

# -----------------------------
# Build + Load model
# -----------------------------
with strategy.scope():
    def build_model(num_classes, l2_weight=1e-4, dropout_rate=0.3):
        base_model = VGG16(include_top=False, input_shape=IMAGE_SIZE + (3,), weights="imagenet")
        x = base_model.output
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dropout(dropout_rate, name="dropout")(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(l2_weight), name="dense_256")(x)
        x = BatchNormalization(name="bn")(x)
        outputs = Dense(num_classes, activation="softmax", dtype="float32", name="pred")(x)
        return keras.Model(inputs=base_model.input, outputs=outputs, name="VGG16_rice62")

    model = build_model(len(RICE_CLASSES))
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_weights(CHECKPOINT_PATH)
            print("✅ Loaded weights from:", CHECKPOINT_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}")
    else:
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_path):
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        # Preprocess
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

        preds = model.predict(image_array, verbose=0)
        idx = int(np.argmax(preds[0]))
        predicted_class = RICE_CLASSES[idx]
        confidence = float(np.max(preds[0]) * 100)

        return predicted_class, confidence

    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    # Example usage
    # predicted_variety, confidence = predict("path/to/rice_image.jpg")
    # print(f"Predicted: {predicted_variety} with {confidence:.2f}% confidence")
    pass
