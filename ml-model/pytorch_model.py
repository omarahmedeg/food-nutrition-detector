"""
PyTorch Food Classification Model
Uses pre-trained fast_food_classifier.pth
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

class FastFoodClassifier:
    def __init__(self, model_path, num_classes=101):
        """
        Initialize PyTorch food classifier
        
        Args:
            model_path: Path to the .pth model file
            num_classes: Number of food classes
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load the checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract information from checkpoint
        if isinstance(checkpoint, dict):
            # Checkpoint contains metadata
            if 'num_classes' in checkpoint:
                self.num_classes = checkpoint['num_classes']
            if 'labels' in checkpoint:
                self.class_names = checkpoint['labels']
            else:
                # Default Food-101 class names (fallback)
                self.class_names = [
                    'apple pie','baby back ribs','baklava','beef carpaccio','beef tartare','beet salad','beignets','bibimbap','bread pudding','breakfast burrito','bruschetta','caesar salad','cannoli','caprese salad','carrot cake','ceviche','cheesecake','cheese plate','chicken curry','chicken quesadilla','chicken wings','chocolate cake','chocolate mousse','churros','clam chowder','club sandwich','crab cakes','creme brulee','croque madame','cup cakes','deviled eggs','donuts','dumplings','edamame','eggs benedict','escargots','falafel','filet mignon','fish and chips','foie gras','french fries','french onion soup','french toast','fried calamari','fried rice','frozen yogurt','garlic bread','gnocchi','greek salad','grilled cheese sandwich','grilled salmon','guacamole','gyoza','hamburger','hot and sour soup','hot dog','huevos rancheros','hummus','ice cream','lasagna','lobster bisque','lobster roll sandwich','macaroni and cheese','macarons','miso soup','mussels','nachos','omelette','onion rings','oysters','pad thai','paella','pancakes','panna cotta','peking duck','pho','pizza','pork chop','poutine','prime rib','pulled pork sandwich','ramen','ravioli','red velvet cake','risotto','samosa','sashimi','scallops','seaweed salad','shrimp and grits','spaghetti bolognese','spaghetti carbonara','spring rolls','steak','strawberry shortcake','sushi','tacos','takoyaki','tiramisu','tuna tartare','waffles'
                ]
            
            # Define the model architecture (ResNet50)
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            # Old format - direct state dict
            # Provide Food-101 default class names as fallback
            self.class_names = [
                'apple pie','baby back ribs','baklava','beef carpaccio','beef tartare','beet salad','beignets','bibimbap','bread pudding','breakfast burrito','bruschetta','caesar salad','cannoli','caprese salad','carrot cake','ceviche','cheesecake','cheese plate','chicken curry','chicken quesadilla','chicken wings','chocolate cake','chocolate mousse','churros','clam chowder','club sandwich','crab cakes','creme brulee','croque madame','cup cakes','deviled eggs','donuts','dumplings','edamame','eggs benedict','escargots','falafel','filet mignon','fish and chips','foie gras','french fries','french onion soup','french toast','fried calamari','fried rice','frozen yogurt','garlic bread','gnocchi','greek salad','grilled cheese sandwich','grilled salmon','guacamole','gyoza','hamburger','hot and sour soup','hot dog','huevos rancheros','hummus','ice cream','lasagna','lobster bisque','lobster roll sandwich','macaroni and cheese','macarons','miso soup','mussels','nachos','omelette','onion rings','oysters','pad thai','paella','pancakes','panna cotta','peking duck','pho','pizza','pork chop','poutine','prime rib','pulled pork sandwich','ramen','ravioli','red velvet cake','risotto','samosa','sashimi','scallops','seaweed salad','shrimp and grits','spaghetti bolognese','spaghetti carbonara','spring rolls','steak','strawberry shortcake','sushi','tacos','takoyaki','tiramisu','tuna tartare','waffles'
            ]
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Number of classes: {self.num_classes}")
        print(f"✓ Classes: {', '.join(self.class_names[:6])}...")
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image, top_k=5):
        """
        Predict food class from image
        
        Args:
            image: PIL Image or path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, confidence)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.class_names[idx.item()]
            confidence = prob.item()
            predictions.append({
                'class': class_name,
                'confidence': confidence
            })
        
        return predictions
    
    def predict_from_array(self, img_array, top_k=5):
        """
        Predict from numpy array
        
        Args:
            img_array: Numpy array (H, W, 3) in range [0, 255]
            top_k: Number of top predictions
            
        Returns:
            List of prediction dictionaries
        """
        # Convert numpy array to PIL Image
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        image = Image.fromarray(img_array)
        return self.predict(image, top_k)
