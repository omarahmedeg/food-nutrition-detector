"""
Flask API Server for Food Classification
Uses PyTorch pre-trained model (food101_classifier.pth)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from PIL import Image

# Add ml-model directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml-model'))

from pytorch_model import FastFoodClassifier
from nutrition_data_food101 import get_nutrition_info

print("✓ Using Food-101 nutrition database")

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js

# Global variable for model
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml-model', 'saved_models', 'food101_classifier.pth')

def load_model():
    """Load the trained PyTorch model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = FastFoodClassifier(MODEL_PATH, num_classes=101)
            print("✓ PyTorch model loaded successfully!")
            print(f"✓ Model path: {MODEL_PATH}")
            print(f"✓ Number of classes: {len(model.class_names)}")
            return True
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}")
            print("Please ensure food101_classifier.pth is in the ml-model/saved_models directory")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict food class from uploaded image
    Expects multipart/form-data with 'image' file
    """
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure food101_classifier.pth exists.'
            }), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        img = Image.open(file.stream).convert('RGB')
        
        # Make prediction using PyTorch model
        predictions = model.predict(img, top_k=5)
        
        # Get top prediction
        top_prediction = predictions[0]
        predicted_class = top_prediction['class']
        confidence = top_prediction['confidence']
        
        # Get nutrition information
        nutrition = get_nutrition_info(predicted_class)
        
        # Format top predictions
        top_predictions = []
        for pred in predictions[:3]:
            top_predictions.append({
                'class': pred['class'],
                'confidence': pred['confidence'],
                'nutrition': get_nutrition_info(pred['class'])
            })
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'food_class': predicted_class,
                'confidence': float(confidence),
                'display_name': predicted_class.replace('_', ' ').title()
            },
            'nutrition': nutrition,
            'top_predictions': top_predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available food classes"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        return jsonify({
            'classes': model.class_names,
            'total': len(model.class_names)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("FOOD NUTRITION DETECTOR API SERVER")
    print("=" * 70)
    
    # Get port from environment variable (for Cloud Run compatibility)
    port = int(os.environ.get('PORT', 5000))
    print(f"\nPort: {port}")
    
    # Check if model file exists before attempting to load
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model file not found at {MODEL_PATH}")
        print("\nPlease ensure the model file exists:")
        print("  1. Train the model: cd ml-model && python train.py")
        print("  2. Or download the pre-trained model")
        print("  3. Place food101_classifier.pth in ml-model/saved_models/")
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    if load_model():
        print(f"✓ Model ready with {len(model.class_names)} classes")
        print(f"✓ Classes: {', '.join(model.class_names[:5])}...")
        
        print("\n" + "=" * 70)
        print("Starting Flask server...")
        print(f"API will be available at: http://localhost:{port}")
        print("=" * 70 + "\n")
        
        # Use production-ready server for Cloud Run
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        print("\n❌ Cannot start server without model.")
        print("\nPlease train the model first:")
        print("  cd ml-model")
        print("  python train.py")
        # Exit with error code if model fails to load
        sys.exit(1)
