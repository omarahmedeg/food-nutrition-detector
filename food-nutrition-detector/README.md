# Food Nutrition Detector 🍔

A complete food classification and nutrition detection system using Machine Learning and Next.js.

![Project Banner](https://img.shields.io/badge/AI-Food%20Classification-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Next.js](https://img.shields.io/badge/Next.js-15-black) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)

## 📋 Overview

This project allows users to upload food images and instantly get detailed nutrition information including:

- **Calories** (kcal)
- **Protein** (g)
- **Carbohydrates** (g)
- **Fat** (g)
- **Fiber** (g)
- **Sugar** (g)

## 🏗️ Project Structure

```
food-nutrition-detector/
├── ml-model/                   # Machine Learning backend
│   ├── model.py               # CNN model architecture
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   ├── nutrition_data.py      # Nutrition database
│   ├── api_server.py          # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── api_requirements.txt   # API dependencies
│   ├── data/                  # Dataset folder
│   │   ├── train/
│   │   └── validation/
│   └── saved_models/          # Trained models (.h5 & .joblib)
│
└── web-app/                    # Next.js frontend
    ├── app/
    │   ├── page.tsx           # Main page
    │   └── layout.tsx
    ├── components/
    │   ├── ImageUpload.tsx    # Image upload component
    │   └── ResultsDisplay.tsx # Results display component
    ├── lib/
    │   └── api.ts             # API utilities
    ├── types/
    │   └── index.ts           # TypeScript types
    └── package.json
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- pip (Python package manager)
- npm or yarn

### Step 1: Set Up the ML Model

```bash
cd ml-model

# Install Python dependencies
pip install -r requirements.txt
pip install -r api_requirements.txt

# Prepare your dataset in this structure:
# data/
#   train/
#     pizza/
#     burger/
#     ...
#   validation/
#     pizza/
#     burger/
#     ...

# Train the model
python train.py --epochs 20

# Test prediction (optional)
python predict.py path/to/test/image.jpg
```

**Dataset Options:**

1. Download Food-101 from [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)
2. Create your own dataset with food photos
3. Use the sample data generator (for testing only)

### Step 2: Start the API Server

```bash
cd ml-model

# Start Flask API server
python api_server.py
```

The API will be available at `http://localhost:5000`

### Step 3: Run the Next.js Frontend

```bash
cd web-app

# Install dependencies
npm install

# Start development server
npm run dev
```

The web app will be available at `http://localhost:3000`

## 🎯 Usage

1. **Open** the web app at `http://localhost:3000`
2. **Upload** a food image by dragging and dropping or clicking to select
3. **Click** "Analyze Food" button
4. **View** detailed nutrition information instantly!

## 🧠 Model Architecture

### Transfer Learning Approach (Recommended)

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers:** Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Softmax
- **Input Size:** 224x224x3 RGB images
- **Optimizer:** Adam (learning rate = 0.001)

### Features:

- Data augmentation (rotation, shift, flip, zoom, shear)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- Batch normalization

## 📊 Supported Food Classes

Currently supports 25+ food items:

- Pizza, Burger, Hot Dog, Tacos, Burrito, Nachos
- Sushi, Ramen, Fried Rice, Dumplings, Spring Rolls
- Steak, Grilled Salmon, Chicken Wings
- French Fries, Spaghetti, Bread
- Ice Cream, Cheesecake, Chocolate Cake, Donuts, Pancakes, Waffles
- Caesar Salad, Omelette, Apple Pie

## 🔌 API Endpoints

### Health Check

```http
GET /api/health
```

### Predict Food

```http
POST /api/predict
Content-Type: multipart/form-data

{
  "image": <file>
}
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "food_class": "pizza",
    "confidence": 0.95,
    "display_name": "Pizza"
  },
  "nutrition": {
    "calories": 266,
    "protein": 11.4,
    "carbs": 33.0,
    "fat": 10.4,
    "fiber": 2.3,
    "sugar": 3.8
  },
  "top_predictions": [...]
}
```

## 🛠️ Technologies Used

### Backend

- **TensorFlow/Keras** - Deep learning framework
- **Flask** - API server
- **Joblib** - Model serialization
- **Pillow** - Image processing
- **NumPy** - Numerical operations

### Frontend

- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **React Dropzone** - File upload
- **Axios** - HTTP client

## 📈 Model Performance Tips

1. **Collect More Data:** 100+ images per class for better accuracy
2. **Balance Classes:** Ensure equal samples per class
3. **Data Augmentation:** Helps with limited datasets
4. **Fine-tuning:** Unfreeze base layers after initial training
5. **GPU Training:** Significantly faster training

## 🎨 Customization

### Add More Food Classes

1. Add images to `ml-model/data/train/new_food_class/`
2. Add nutrition info to `ml-model/nutrition_data.py`:

```python
'new_food_class': {
    'calories': 200,
    'protein': 10.0,
    'carbs': 25.0,
    'fat': 8.0,
    'fiber': 2.0,
    'sugar': 5.0
}
```

3. Retrain the model: `python train.py`

### Change Model Architecture

Edit `ml-model/model.py` to modify the CNN architecture or use different pre-trained models (ResNet, EfficientNet, etc.)

## 🐛 Troubleshooting

**Problem:** Model not loading

- **Solution:** Make sure you've trained the model first: `python train.py`

**Problem:** API connection failed

- **Solution:** Ensure Flask server is running on port 5000

**Problem:** Low accuracy

- **Solution:** Add more training data or increase epochs

**Problem:** Out of memory during training

- **Solution:** Reduce batch_size in `train.py`

**Problem:** Slow predictions

- **Solution:** Use GPU or reduce image size

## 📝 Environment Variables

Create a `.env.local` file in the `web-app` folder:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## 🚀 Deployment

### Deploy ML Model

- Use **Google Cloud Run**, **AWS Lambda**, or **Heroku**
- Make sure to include all dependencies
- Update CORS settings for production

### Deploy Next.js App

- Deploy to **Vercel**, **Netlify**, or any hosting service
- Update `NEXT_PUBLIC_API_URL` to production API URL

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

Created as part of a nutrition and diet awareness initiative.

## 🙏 Acknowledgments

- Food-101 Dataset
- TensorFlow/Keras Documentation
- MobileNetV2 Architecture
- Next.js Team

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

**Made with ❤️ for healthier food choices**
