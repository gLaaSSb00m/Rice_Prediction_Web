from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import warnings
import traceback
from .models import RiceInfo

# Rice classification labels (extended to 20 classes)


RICE_CLASSES = [
    "1_Subol_Lota","2_Bashmoti","3_Ganjiya","4_Shampakatari","5_Katarivog","6_BR28","7_BR29", "8_Paijam", "9_Bashful",
    "10_Lal_Aush","11_Jirashail","12_Gutisharna","13_Red_Cargo","14_Najirshail","15_Katari_Polao","16_Lal_Biroi",
    "17_Chinigura_Polao","18_Amon","19_Shorna5","20_Lal_Binni"
]
# Note: Replace 'Unknown1' to 'Unknown6' with the correct 6 additional classes if available.

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
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
@never_cache
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

            # Fetch rice info from database
            try:
                rice_info_obj = RiceInfo.objects.get(variety_name=predicted_class)
                rice_info = rice_info_obj.info
            except RiceInfo.DoesNotExist:
                rice_info = "No info available for this variety yet."

            return JsonResponse({
                'predicted_variety': predicted_class,
                'rice_info': rice_info,
                'message': f"Predicted Rice Variety: {predicted_class}\nInfo: {rice_info}"
            })

        except Exception as e:
            return render(request, 'prediction/predict.html', {'error': str(e)})

    # GET request → just render empty form with cleared context
    return render(request, 'prediction/predict.html', {
        'predicted_variety': None,
        'rice_info': None,
        'error': None,
        'message': None
    })


def home(request):
    """Render the home page."""
    return render(request, 'prediction/home.html')
