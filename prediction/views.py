import os, warnings, traceback
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
from django.conf import settings
from .models import RiceInfo, RiceModel

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
IMAGE_SIZE = (224, 224)

# Load classes and model from DB
def load_classes_and_model():
    rice_classes = list(RiceInfo.objects.values_list('variety_name', flat=True))
    active_model = RiceModel.objects.filter(is_active=True).first()
    return rice_classes, active_model

RICE_CLASSES, ACTIVE_MODEL = load_classes_and_model()

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

    if ACTIVE_MODEL and os.path.exists(ACTIVE_MODEL.model_file.path):
        try:
            model.load_weights(ACTIVE_MODEL.model_file.path)
            print("✅ Loaded weights from:", ACTIVE_MODEL.model_file.path)
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}")
    else:
        print("[ERROR] No active model or file not found")

# -----------------------------
# Prediction View
# -----------------------------
@csrf_exempt
@never_cache
def predict(request):
    warnings.filterwarnings("ignore", category=UserWarning)

    if request.method == "POST":
        try:
            image_file = request.FILES.get("rice_image")
            if not image_file:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Preprocess
            image = Image.open(image_file).convert("RGB")

            # Save the uploaded image to fixed location
            path = os.path.join(settings.MEDIA_ROOT, 'predictions', 'current.jpg')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)

            image = image.resize(IMAGE_SIZE)
            image_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

            preds = model.predict(image_array, verbose=0)
            idx = int(np.argmax(preds[0]))
            predicted_class = RICE_CLASSES[idx]
            confidence = float(np.max(preds[0]) * 100)

            rice_info_obj = RiceInfo.objects.filter(variety_name=predicted_class).first()
            rice_info = rice_info_obj.info if rice_info_obj else "No info available."

            # Close the image
            image.close()

            # Delete the uploaded image file if it's a temporary file
            if hasattr(image_file, 'temporary_file_path'):
                try:
                    os.remove(image_file.temporary_file_path())
                except OSError:
                    pass  # Ignore if deletion fails

            return JsonResponse({
                "predicted_variety": predicted_class,
                "confidence": confidence,
                "rice_info": rice_info,
                "message": f"Predicted Rice Variety: {predicted_class} ({confidence:.2f}% confidence)"
            })

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "prediction/predict.html")

def home(request):
    return render(request, "prediction/home.html")

# New endpoint to get active model
def get_model(request):
    active_model = RiceModel.objects.filter(is_active=True).first()
    if not active_model or not active_model.tflite_file or not os.path.exists(active_model.tflite_file.path):
        return JsonResponse({"error": "No active model available"}, status=404)
    try:
        with open(active_model.tflite_file.path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{active_model.name}.tflite"'
            return response
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# New endpoint to get rice info
def get_rice_info(request):
    rice_infos = list(RiceInfo.objects.values('variety_name', 'info', 'updated_at'))
    return JsonResponse({"rice_infos": rice_infos})
