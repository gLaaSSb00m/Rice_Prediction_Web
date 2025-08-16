# Rice Classification Web Application

A Django-based web application for classifying rice varieties using a ConvNeXtBase deep learning model. The application can classify 15 different types of rice with high accuracy.

## Screenshots

### Home Page
![Home Page](WhatsApp%20Image%202025-08-15%20at%2009.48.49_afa313f2.jpg)

### Prediction Interface
![Prediction Interface](WhatsApp%20Image%202025-08-16%20at%2011.19.22_f0ccaa09.jpg)

### Results Display
![Results Display](WhatsApp%20Image%202025-08-16%20at%2011.19.39_f4fef0af.jpg)

## Features

- **Image Upload**: Users can upload rice grain images for classification
- **Real-time Prediction**: Instant classification results using ConvNeXtBase model
- **Confidence Scores**: Displays prediction confidence for each rice type
- **15 Rice Types**: Supports classification of 15 different rice varieties
- **Responsive Design**: Works on desktop and mobile devices

## Installation

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

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Click on "Predict Rice Type"
3. Upload a rice grain image
4. View the classification results with confidence scores

## Model Information

- **Architecture**: ConvNeXtBase
- **Training Dataset**: 15 rice varieties
- **Accuracy**: High accuracy on test dataset
- **Model File**: `ConvNeXtBase_Rice_Classification_lr0001ep25bt32_Adam_CCE.h5`

## Supported Rice Types

The application can classify the following rice varieties:
- [List of 15 rice types will be populated based on your model's classes]

## Technical Stack

- **Backend**: Django 4.x
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: PIL/Pillow
- **Database**: SQLite (default)

## Project Structure

```
Rice_Prediction/
├── rice_prediction/          # Django project settings
├── prediction/              # Main app for rice classification
│   ├── views.py            # Prediction logic
│   ├── models.py           # Database models
│   └── urls.py             # URL configurations
├── templates/              # HTML templates
├── static/                # CSS, JS, images
├── requirements.txt       # Python dependencies
├── manage.py             # Django management script
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
2. Update the class mapping in `views.py`
3. Update the frontend to display new types

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
