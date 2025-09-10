# Rice Classification System

A comprehensive rice variety classification system with both web and mobile applications. Uses a ConvNeXtBase deep learning model to classify 20 different types of rice with high accuracy.

## 🚀 Features

### Web Application
- **Image Upload**: Users can upload rice grain images for classification
- **Real-time Prediction**: Instant classification results using ConvNeXtBase model
- **Confidence Scores**: Displays prediction confidence for each rice type
- **20 Rice Types**: Supports classification of 20 different rice varieties
- **Responsive Design**: Works on desktop and mobile devices
- **REST API**: Provides API endpoints for mobile app integration

### Mobile Application (Flutter)
- **Camera Integration**: Take photos directly from camera
- **Gallery Access**: Select images from device gallery
- **Real-time Prediction**: Instant classification using backend API
- **Cross-platform**: Works on both Android and iOS
- **Offline Support**: Basic functionality works without internet
- **Modern UI**: Material Design 3 with intuitive interface

## Installation

### Web Application

1. Clone the repository:
```bash
git clone <repository-url>
cd Rice_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Start the development server:
```bash
python manage.py runserver
```

### Mobile Application

1. Navigate to the mobile app directory:
```bash
cd rice_prediction_mobile
```

2. Get Flutter dependencies:
```bash
flutter pub get
```

3. Run the app on an emulator or device:
```bash
flutter run
```

## Usage

### Web

1. Open your browser and navigate to `http://localhost:8000`
2. Click on "Predict Rice Type"
3. Upload a rice grain image
4. View the classification results with confidence scores

### Mobile

1. Launch the app on your device or emulator
2. Use the camera or gallery to select a rice grain image
3. Tap "Predict Rice Variety"
4. View the prediction results and rice information

## Model Information

- **Architecture**: ConvNeXtBase
- **Training Dataset**: 20 rice varieties
- **Accuracy**: High accuracy on test dataset
- **Model File**: `ConvNeXtBase_Rice_Classification_lr0001ep25bt32_Adam_CCE.h5`

## Supported Rice Types

The application can classify the following rice varieties:
- 1_Subol_Lota
- 2_Bashmoti
- 3_Ganjiya
- 4_Shampakatari
- 5_Katarivog
- 6_BR28
- 7_BR29
- 8_Paijam
- 9_Bashful
- 10_Lal_Aush
- 11_Jirashail
- 12_Gutisharna
- 13_Red_Cargo
- 14_Najirshail
- 15_Katari_Polao
- 16_Lal_Biroi
- 17_Chinigura_Polao
- 18_Amon
- 19_Shorna5
- 20_Lal_Binni

## Technical Stack

- **Backend**: Django 5.x, Django REST Framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Mobile**: Flutter (Dart)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: PIL/Pillow
- **Database**: SQLite (default)

## Project Structure

```
Rice_Prediction/
├── rice_prediction/          # Django project settings
├── prediction/               # Main app for rice classification
│   ├── views.py             # Prediction logic (web)
│   ├── api_views.py         # API views for mobile app
│   ├── serializers.py       # DRF serializers
│   ├── models.py            # Database models
│   └── urls.py              # URL configurations
├── templates/               # HTML templates
├── static/                  # CSS, JS, images
├── rice_prediction_mobile/  # Flutter mobile app
│   ├── lib/                 # Flutter source code
│   ├── android/             # Android platform code
│   └── ios/                 # iOS platform code
├── requirements.txt         # Python dependencies
├── manage.py                # Django management script
└── README.md
```

## Development

### Running Tests
```bash
python manage.py test prediction
```

### Adding New Rice Types
To add support for new rice types, you'll need to:
1. Retrain the model with new data
2. Update the class mapping in `views.py` and `api_views.py`
3. Update the frontend and mobile app to display new types

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
