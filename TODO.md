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
