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

# Rice classification labels (extended to 20 classes)
RICE_CLASSES = [
    "1_Subol_Lota","2_Bashmoti","3_Ganjiya","4_Shampakatari","5_Katarivog","6_BR28","7_BR29", "8_Paijam", "9_Bashful",
    "10_Lal_Aush","11_Jirashail","12_Gutisharna","13_Red_Cargo","14_Najirshail","15_Katari_Polao","16_Lal_Biroi",
    "17_Chinigura_Polao","18_Amon","19_Shorna5","20_Lal_Binni"
]

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
