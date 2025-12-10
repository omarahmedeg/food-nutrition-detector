# Food Nutrition Detector - ML Model

This folder contains the machine learning model for food classification and nutrition detection.

## 📋 Overview

The model uses a Convolutional Neural Network (CNN) to classify food items from images and provides detailed nutrition information including calories, protein, carbs, fat, fiber, and sugar content.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your food images in this structure:

```
data/
├── train/
│   ├── pizza/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── burger/
│   ├── sushi/
│   └── ...
└── validation/
    ├── pizza/
    ├── burger/
    └── ...
```

**Dataset Options:**

- Download Food-101 from [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)
- Create your own by taking photos of different food items
- Use the sample data generator (for testing only)

### 3. Train the Model

```bash
# Using transfer learning (recommended)
python train.py --epochs 20

# Using simple CNN
python train.py --epochs 30 --simple-cnn
```

### 4. Make Predictions

```bash
# Predict from local image
python predict.py path/to/image.jpg

# Predict from URL
python predict.py https://example.com/food-image.jpg
```

## 📁 Files Description

- **model.py** - CNN model architecture and training logic
- **train.py** - Training script with data augmentation
- **predict.py** - Prediction script for single images
- **nutrition_data.py** - Nutrition database for 25+ food items
- **requirements.txt** - Python dependencies

## 🧠 Model Architecture

### Transfer Learning (Default)

- Base: MobileNetV2 (pre-trained on ImageNet)
- Custom layers: Dense(512) → Dropout → Dense(256) → Dropout → Output
- Input: 224x224x3 RGB images
- Optimizer: Adam (lr=0.001)

### Simple CNN (Alternative)

- 4 Convolutional blocks with BatchNormalization
- MaxPooling and Dropout for regularization
- Dense layers: 512 → 256 → Output

## 📊 Features

- **Data Augmentation**: Rotation, shift, flip, zoom, shear
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Model Checkpointing**: Saves best model during training
- **Joblib Export**: Easy model serialization

## 🥗 Supported Food Classes

Currently supports 25+ food items including:

- Pizza, Burger, Hot Dog
- Sushi, Ramen, Fried Rice
- Steak, Salmon, Chicken Wings
- Ice Cream, Cheesecake, Donuts
- And more...

## 💾 Model Export

The trained model is saved in two formats:

- **food_model.h5** - Keras model with weights
- **food_classifier.joblib** - Complete configuration

## 📈 Usage in Python

```python
from model import FoodClassificationModel
from nutrition_data import get_nutrition_info

# Load model
model = FoodClassificationModel.load_model('saved_models/food_classifier.joblib')

# Predict
food_class, confidence, _ = model.predict('image.jpg')

# Get nutrition info
nutrition = get_nutrition_info(food_class)
print(f"Calories: {nutrition['calories']} kcal")
```

## 🎯 Performance Tips

1. **More Data**: Collect 100+ images per class for better accuracy
2. **Transfer Learning**: Use pre-trained models (MobileNetV2, ResNet50)
3. **Data Augmentation**: Helps with limited datasets
4. **Class Balance**: Ensure equal samples per class
5. **Fine-tuning**: Unfreeze base layers after initial training

## 🔧 Customization

### Add More Food Classes

1. Add images to `data/train/new_food_class/`
2. Add nutrition info to `nutrition_data.py`
3. Retrain the model

### Change Input Size

```python
model = FoodClassificationModel(img_height=299, img_width=299)
```

## 📝 Notes

- Model performance depends on dataset quality
- Sample synthetic data is for testing only
- Use real food images for production
- Consider fine-tuning for specific cuisines

## 🐛 Troubleshooting

**Out of Memory Error**: Reduce batch_size in train.py
**Low Accuracy**: Add more training data or increase epochs
**Slow Training**: Use GPU or reduce image size

## 📚 References

- TensorFlow/Keras Documentation
- Food-101 Dataset Paper
- MobileNetV2 Architecture
