import os, warnings, traceback
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from xgboost import XGBClassifier
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
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

# Load classes and models from DB
def load_classes_and_models():
    rice_classes = list(RiceInfo.objects.values_list('variety_name', flat=True))
    models = RiceModel.objects.filter(is_active=True)
    vgg_model = models.filter(name='VGG16 Rice Classifier').first()
    mobile_model = models.filter(name='MobileNetV2 Rice Classifier').first()
    xgb_model = models.filter(name='XGBoost Meta Model').first()
    vit_model = models.filter(name='ViT Rice Classifier').first()
    return rice_classes, vgg_model, mobile_model, xgb_model, vit_model

RICE_CLASSES, VGG_MODEL, MOBILE_MODEL, XGB_MODEL, VIT_MODEL = load_classes_and_models()

# -----------------------------
# Build + Load models
# -----------------------------
with strategy.scope():
    def build_vgg16_feature_extractor():
        base = VGG16(weights=None, include_top=False, input_shape=(224,224,3))
        x = GlobalAveragePooling2D()(base.output)
        model = keras.Model(inputs=base.input, outputs=x)
        if VGG_MODEL and os.path.exists(VGG_MODEL.model_file.path):
            model.load_weights(VGG_MODEL.model_file.path)
            print("✅ Loaded VGG16 weights from:", VGG_MODEL.model_file.path)
        else:
            print("[ERROR] VGG16 model file not found")
        return model

    def build_mobilenetv2_feature_extractor():
        base = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
        x = GlobalAveragePooling2D()(base.output)
        model = keras.Model(inputs=base.input, outputs=x)
        if MOBILE_MODEL and os.path.exists(MOBILE_MODEL.model_file.path):
            model.load_weights(MOBILE_MODEL.model_file.path)
            print("✅ Loaded MobileNetV2 weights from:", MOBILE_MODEL.model_file.path)
        else:
            print("[ERROR] MobileNetV2 model file not found")
        return model

    def build_vgg16_classifier(num_classes, l2_weight=1e-4, dropout_rate=0.3):
        base_model = VGG16(include_top=False, input_shape=IMAGE_SIZE + (3,), weights="imagenet")
        x = base_model.output
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dropout(dropout_rate, name="dropout")(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(l2_weight), name="dense_256")(x)
        x = BatchNormalization(name="bn")(x)
        outputs = Dense(num_classes, activation="softmax", dtype="float32", name="pred")(x)
        model = keras.Model(inputs=base_model.input, outputs=outputs, name="VGG16_rice62")
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        if VGG_MODEL and os.path.exists(VGG_MODEL.model_file.path):
            model.load_weights(VGG_MODEL.model_file.path)
            print("✅ Loaded VGG16 classifier weights from:", VGG_MODEL.model_file.path)
        else:
            print("[ERROR] VGG16 classifier model file not found")
        return model

    vgg_feature_extractor = build_vgg16_feature_extractor()
    mobile_feature_extractor = build_mobilenetv2_feature_extractor()
    vgg_classifier = build_vgg16_classifier(len(RICE_CLASSES))

    # Load XGBoost model
    xgb_classifier = None
    if XGB_MODEL and os.path.exists(XGB_MODEL.model_file.path):
        xgb_classifier = XGBClassifier()
        xgb_classifier.load_model(XGB_MODEL.model_file.path)
        print("✅ Loaded XGBoost model from:", XGB_MODEL.model_file.path)
    else:
        print("[ERROR] XGBoost model file not found")

# Load ViT model outside strategy scope since it's PyTorch
vit_classifier = None
if VIT_MODEL and os.path.exists(VIT_MODEL.model_file.path):
    vit_classifier = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=len(RICE_CLASSES),
        ignore_mismatched_sizes=True
    )
    vit_classifier.load_state_dict(torch.load(VIT_MODEL.model_file.path, map_location='cpu'))
    vit_classifier.eval()
    print("✅ Loaded ViT model from:", VIT_MODEL.model_file.path)
else:
    print("[ERROR] ViT model file not found")

# ViT transform
vit_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Helper Functions
# -----------------------------
def extract_features(img_path, model, preprocess_func):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    feat = model.predict(img_array, verbose=0)
    return feat.flatten()

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
            model_type = request.POST.get("model_type", "vgg16")  # Default to vgg16
            if not image_file:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Preprocess
            image = Image.open(image_file).convert("RGB")

            # Save the uploaded image to fixed location
            path = os.path.join(settings.MEDIA_ROOT, 'predictions', 'current.jpg')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)

            if model_type == "ensemble":
                # Ensemble prediction
                feat_vgg = extract_features(path, vgg_feature_extractor, vgg_preprocess)
                feat_mobile = extract_features(path, mobile_feature_extractor, mobilenet_preprocess)
                stacked_feat = np.hstack([feat_vgg, feat_mobile]).reshape(1, -1)  # Reshape to (1, num_features)
                pred_index = xgb_classifier.predict(stacked_feat)[0]
                confidence = float(xgb_classifier.predict_proba(stacked_feat).max() * 100)
            elif model_type == "vit":
                # ViT prediction
                if vit_classifier is None:
                    return JsonResponse({"error": "ViT model not loaded"}, status=500)
                img_tensor = vit_transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = vit_classifier(img_tensor)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    pred_index = torch.argmax(probs, dim=1).item()
                    confidence = float(probs[0][pred_index].item() * 100)
            else:
                # VGG16 prediction
                image_resized = image.resize(IMAGE_SIZE)
                image_array = np.expand_dims(np.array(image_resized, dtype=np.float32) / 255.0, axis=0)
                preds = vgg_classifier.predict(image_array, verbose=0)
                pred_index = int(np.argmax(preds[0]))
                confidence = float(np.max(preds[0]) * 100)

            predicted_class = RICE_CLASSES[pred_index]
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


