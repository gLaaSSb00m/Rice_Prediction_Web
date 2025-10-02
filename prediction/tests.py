import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
import os

NUM_CLASSES = 62
INPUT_SHAPE = (224, 224, 3)

# Recreate the exact architecture
base_model = keras.applications.VGG16(include_top=False, input_shape=INPUT_SHAPE, weights="imagenet")
x = base_model.output
x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
x = keras.layers.Dropout(0.3, name="dropout")(x)
x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_256")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="pred")(x)
model = keras.Model(inputs=base_model.input, outputs=outputs, name="VGG16_rice62")

# Load your trained weights file
MODEL_PATH = os.path.join("models", "best_VGG16_stage2.weights.h5")
if os.path.exists(MODEL_PATH):
    try:
        model.load_weights(MODEL_PATH)
        print(f"✅ Loaded weights from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
else:
    print(f"[ERROR] Checkpoint not found at {MODEL_PATH}")

loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

# Check that the model object exists and is usable
print(type(model))

# Print a one-line confirmation of the architecture
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Print the full layer-by-layer summary
print(model.summary())





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



# Preprocess image
image = Image.open('SubolLota_1_019.jpg').convert('RGB').resize((224, 224))
image_array = np.array(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
predictions = model.predict(image_array)
predicted_idx = int(np.argmax(predictions[0]))
predicted_class = RICE_CLASSES[predicted_idx]
confidence = float(np.max(predictions[0]) * 100)

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
