from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import traceback
from .models import RiceInfo
from .serializers import RiceInfoSerializer

# Rice classification labels (62 classes)
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

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_VGG16_stage2.weights.h5')

# Load model at startup
model = None
try:
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    # Recreate the exact architecture
    base_model = keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_256")(x)
    x = keras.layers.BatchNormalization(name="bn")(x)
    outputs = keras.layers.Dense(len(RICE_CLASSES), activation="softmax", dtype="float32", name="pred")(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs, name="VGG16_rice62")

    if os.path.exists(MODEL_PATH):
        try:
            model.load_weights(MODEL_PATH)
            print(f"[INFO] Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}")
    else:
        print(f"[ERROR] Checkpoint not found at {MODEL_PATH}")

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    print("[INFO] Model loaded and compiled successfully")
except Exception as e:
    print(f"[ERROR] Model loading failed: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")

@method_decorator(csrf_exempt, name='dispatch')
class PredictAPIView(APIView):
    """API view for rice image prediction."""

    def post(self, request):
        try:
            rice_image = request.FILES.get('rice_image')
            if not rice_image:
                return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Preprocess
            image = Image.open(rice_image).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            if model is None:
                return Response({'error': 'Model not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

            return Response({
                'predicted_variety': predicted_class,
                'rice_info': rice_info,
                'confidence': confidence,
                'message': f"Predicted Rice Variety: {predicted_class}\nInfo: {rice_info}"
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RiceInfoListAPIView(APIView):
    """API view to list all rice varieties."""

    def get(self, request):
        rice_infos = RiceInfo.objects.all()
        serializer = RiceInfoSerializer(rice_infos, many=True)
        return Response(serializer.data)
