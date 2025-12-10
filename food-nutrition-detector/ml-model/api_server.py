"""
Flask API Server for Food Classification
Uses PyTorch pre-trained model (food101_classifier.pth)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image

from pytorch_model import FastFoodClassifier
from nutrition_data_food101 import get_nutrition_info

print("✓ Using Food-101 nutrition database")

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js

# Global variable for model
model = None

# Try multiple possible model paths
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), 'saved_models', 'food101_classifier.pth'),
    os.path.join(os.path.dirname(__file__), 'saved_models', 'fast_food_classifier.pth'),
    '/workspace/saved_models/fast_food_classifier.pth',
    '/workspace/ml-model/saved_models/food101_classifier.pth'
]

def load_model():
    """Load the trained PyTorch model"""
    global model
    
    # Try to find the model file
    MODEL_PATH = None
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    
    try:
        if MODEL_PATH:
            model = FastFoodClassifier(MODEL_PATH, num_classes=101)
            print("✓ PyTorch model loaded successfully!")
            print(f"✓ Model path: {MODEL_PATH}")
            print(f"✓ Number of classes: {len(model.class_names)}")
            return True
        else:
            print(f"⚠️  Model not found in any of these locations:")
            for path in POSSIBLE_MODEL_PATHS:
                print(f"   - {path}")
            print("⚠️  Starting server without model (health check will work)")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Food Nutrition Detector API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict',
            'classes': '/api/classes'
        }
    })

if __name__ == '__main__':
    print("=" * 70)
    print("FOOD NUTRITION DETECTOR API SERVER")
    print("=" * 70)
    
    # Load model (but don't fail if it's not available)
    print("\nLoading model...")
    load_model()
    
    # Get port from environment variable (Cloud Run provides this)
    port = int(os.environ.get('PORT', 8080))
    
    print("\n" + "=" * 70)
    print("Starting Flask server...")
    print(f"API will be available on port: {port}")
    print("=" * 70 + "\n")
    
    # Start server (no debug mode in production)
    app.run(host='0.0.0.0', port=port, debug=False)
