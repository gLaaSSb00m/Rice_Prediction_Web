import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras

NUM_CLASSES = 20
INPUT_SHAPE = (224, 224, 3)

# Recreate the exact architecture
base = keras.applications.convnext.ConvNeXtBase(
    include_top=False, weights=None, input_shape=INPUT_SHAPE
)
for l in base.layers:
    l.trainable = False

x = keras.layers.Flatten()(base.output)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs=base.input, outputs=out)

# Load your trained weights file (change the filename if different)
model.load_weights("ConvNeXtBase_Rice_Classification_lr0001ep25bt32_Adam_CCE.h5")

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Check that the model object exists and is usable
print(type(model))

# Print a one-line confirmation of the architecture
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Print the full layer-by-layer summary
print(model.summary())





RICE_CLASSES = [
    "1_Subol_Lota","2_Bashmoti","3_Ganjiya","4_Shampakatari","5_Katarivog","6_BR28","7_BR29", "8_Paijam", "9_Bashful",
    "10_Lal_Aush","11_Jirashail","12_Gutisharna","13_Red_Cargo","14_Najirshail","15_Katari_Polao","16_Lal_Biroi",
    "17_Chinigura_Polao","18_Amon","19_Shorna5","20_Lal_Binni"
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