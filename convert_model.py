import tensorflow as tf
from tensorflow import keras
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_VGG16_stage2.weights.h5')
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'rice_model.tflite')

# Rice classes
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

# Recreate the model architecture
base_model = keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
x = base_model.output
x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
x = keras.layers.Dropout(0.3, name="dropout")(x)
x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_256")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
outputs = keras.layers.Dense(len(RICE_CLASSES), activation="softmax", dtype="float32", name="pred")(x)
model = keras.Model(inputs=base_model.input, outputs=outputs, name="VGG16_rice62")

# Load weights
if os.path.exists(MODEL_PATH):
    try:
        model.load_weights(MODEL_PATH)
        print(f"✅ Loaded weights from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
else:
    print(f"[ERROR] Checkpoint not found at {MODEL_PATH}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {TFLITE_MODEL_PATH}")
