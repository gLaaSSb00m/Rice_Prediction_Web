from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import warnings
import traceback

# Rice classification labels (extended to 20 classes)
RICE_CLASSES = [
    'Subol lota', 'Bashmoti', 'Ganjiya', 'Shampakatari', 'katarivog',
    'BR28', 'BR29', 'Paijam', 'Bashful', 'Lal Aush', 'Jirashail',
    'Gutisharna', 'Red Cargo', 'Najirshail', 'Unknown1', 'Unknown2',
    'Unknown3', 'Unknown4', 'Unknown5', 'Unknown6'
]
# Note: Replace 'Unknown1' to 'Unknown6' with the correct 6 additional classes if available.

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ConvNeXtBase_Rice_Classification_lr0001ep25bt32_Adam_CCE.h5')

# Load model at startup
model = None
try:
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    # Recreate the exact architecture
    base = keras.applications.convnext.ConvNeXtBase(
        include_top=False, weights=None, input_shape=(224, 224, 3)
    )
    for l in base.layers:
        l.trainable = False
    x = keras.layers.Flatten()(base.output)
    out = keras.layers.Dense(len(RICE_CLASSES), activation="softmax")(x)  # 20 classes
    model = keras.Model(inputs=base.input, outputs=out)
    print(f"[INFO] Loading weights from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] Model weights not found: {MODEL_PATH}")
    model.load_weights(MODEL_PATH)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print("[INFO] Model loaded and compiled successfully")
except Exception as e:
    print(f"[ERROR] Model loading failed: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")


@csrf_exempt
def predict(request):
    """Handle rice image prediction requests and render results in the template."""
    warnings.filterwarnings("ignore", category=UserWarning)

    if request.method == 'POST':
        try:
            image_file = request.FILES.get('rice_image')
            if not image_file:
                return render(request, 'prediction/predict.html', {'error': 'No image provided'})

            # Preprocess
            image = Image.open(image_file).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            if model is None:
                return render(request, 'prediction/predict.html', {'error': 'Model not loaded'})

            # Predict
            predictions = model.predict(image_array, verbose=0)
            predicted_idx = int(np.argmax(predictions[0]))
            predicted_class = RICE_CLASSES[predicted_idx]
            confidence = float(np.max(predictions[0]) * 100)

            return render(request, 'prediction/predict.html', {
                # 'message': f"Predicted Class: {predicted_class} with {confidence:.2f}% confidence",
                'predicted_variety': predicted_class,
                'rice_info': "No info available for this variety yet."
            })

        except Exception as e:
            return render(request, 'prediction/predict.html', {'error': str(e)})

    # GET request → just render empty form
    return render(request, 'prediction/predict.html')


def home(request):
    """Render the home page."""
    return render(request, 'prediction/home.html')