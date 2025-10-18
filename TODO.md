# TODO: Integrate ViT Model for Rice Classification

- [x] Update requirements.txt: Add torch, torchvision, transformers for ViT support.
- [x] Modify prediction/views.py: Add ViT model loading and prediction logic as separate model_type "vit".
- [x] Update templates/prediction/predict.html: Add ViT option to model selection dropdown.
- [ ] Test ViT prediction by uploading model file via Django admin and setting as active.
- [ ] Run Django server and verify "vit" model_type works in prediction form.
