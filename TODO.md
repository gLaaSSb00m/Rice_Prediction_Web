# TODO: Make Rice Variety Name and Rice Info Fields Uneditable

## Steps to Complete:
- [x] Fix form field name mismatch: Change 'image' to 'rice_image' in predict.html to match views.py
- [x] Modify Rice Variety Name field: Change to read-only input, populate with predicted_variety
- [x] Modify Rice Info field: Change to read-only textarea, populate with rice_info
- [x] Remove 'required' attribute from variety and info fields
- [x] Update form submission to exclude variety and info from FormData if not needed
- [x] Test the changes by running the Django server and uploading an image

## Dependent Files:
- templates/prediction/predict.html
- prediction/views.py (updated to return JSON for AJAX)

## Followup Steps:
- Run `python manage.py runserver` to test the prediction page
- Upload an image and verify that variety and info are displayed as read-only text
- Ensure no errors in form submission

---

# TODO: Update Mobile App for Local TFLite Prediction

## Steps to Complete:
- [x] Add TFLite dependency to pubspec.yaml
- [x] Update main.dart to load TFLite model in initState
- [x] Replace _predictRice method to use TFLite.runModelOnImage instead of API call
- [x] Remove unused imports (http, http_parser, kIsWeb, image package)
- [x] Create assets/models directory and placeholder for rice_model.tflite
- [x] Update pubspec.yaml to include assets
- [x] Run flutter pub get to install dependencies

## Dependent Files:
- rice_prediction_mobile/pubspec.yaml
- rice_prediction_mobile/lib/main.dart
- rice_prediction_mobile/assets/models/rice_model.tflite (placeholder)

## Followup Steps:
- Replace the placeholder rice_model.tflite with the actual converted TFLite model
- Test the app on a mobile device or emulator
- Ensure permissions are granted and model loads correctly
- Verify predictions work with sample images
