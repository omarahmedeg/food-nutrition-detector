# Backend Inference Example
import torch
from torchvision import transforms
from PIL import Image

# Load model
checkpoint = torch.load('fast_food_classifier.pth', map_location='cpu')
labels = checkpoint['labels']
num_classes = checkpoint['num_classes']

# Recreate model architecture
from torchvision import models
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inference function
def predict(image_path):
    """
    Predict the class of a fast food image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Contains class, confidence, and all probabilities
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'class': labels[predicted.item()],
        'confidence': float(confidence.item()),
        'all_probabilities': {labels[i]: float(probabilities[0][i].item()) for i in range(num_classes)}
    }

# Example usage for Flask/FastAPI
def predict_from_bytes(image_bytes):
    """Predict from image bytes (useful for API endpoints)"""
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'class': labels[predicted.item()],
        'confidence': float(confidence.item()),
        'all_probabilities': {labels[i]: float(probabilities[0][i].item()) for i in range(num_classes)}
    }

# Test the function
# result = predict('path/to/your/image.jpg')
# print(f"Predicted: {result['class']}")
# print(f"Confidence: {result['confidence']:.2%}")
