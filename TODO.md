# TODO: Implement Flutter App Synchronization with Django

## Django Backend Changes
- [x] Add `get_model` endpoint to serve the active ML model file
- [x] Add `get_rice_info` endpoint to serve rice varieties information as JSON
- [x] Update prediction/urls.py to include new endpoints

## Flutter App Changes
- [x] Update pubspec.yaml with new dependencies: connectivity_plus, http, path_provider, shared_preferences
- [x] Modify main.dart to:
  - [x] Add connectivity check
  - [x] Add sync service to download model and rice info when online
  - [x] Change model loading from asset to local file
  - [x] Store and load rice classes from local storage
  - [x] Update prediction logic to use local data

## Testing
- [x] Test Django endpoints
- [x] Test Flutter sync functionality
- [x] Verify offline functionality
