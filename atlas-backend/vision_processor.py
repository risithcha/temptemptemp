import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image
import io
import numpy as np
import easyocr
import threading

# Global variables to cache the model
_model = None
_weights = None
_device = None
_model_lock = threading.Lock()

# Global variable to cache the OCR reader
_ocr_reader = None
_ocr_lock = threading.Lock()

def _load_model():
    """
    Load the object detection model.
    Cached after first call for performance.
    """
    global _model, _weights, _device
    
    if _model is None:
        with _model_lock:
            if _model is None:
                # Use GPU if available, otherwise CPU
                _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load pre-trained model and weights
                _weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                _model = ssdlite320_mobilenet_v3_large(weights=_weights)
                _model.to(_device)
                _model.eval()
                
                print(f"Model loaded successfully on device: {_device}")
    
    return _model, _weights, _device

def _load_ocr_reader():
    """
    Load the OCR reader.
    Cached after first call for performance.
    """
    global _ocr_reader
    
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                # Use GPU if available, otherwise CPU
                use_gpu = torch.cuda.is_available()
                
                # Load OCR reader with English language support
                _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
                
                print(f"OCR reader loaded successfully (GPU: {use_gpu})")
    
    return _ocr_reader

def preload_models():
    """Eagerly load both the detection model and OCR reader at startup."""
    _load_model()
    _load_ocr_reader()


def _generate_scene_description(detected_objects, ocr_text):
    """
    Generate natural language scene description.
    Combines detected objects and OCR text.
    """
    if not detected_objects and not ocr_text:
        return "No objects or text detected in the image."
    
    description_parts = []
    
    # Add object detection results
    if detected_objects:
        object_labels = [obj['label'] for obj in detected_objects]
        
        if len(object_labels) == 1:
            description_parts.append(f"I see a {object_labels[0]}")
        elif len(object_labels) == 2:
            description_parts.append(f"I see a {object_labels[0]} and a {object_labels[1]}")
        else:
            objects_list = ', '.join(object_labels[:-1])
            description_parts.append(f"I see a {objects_list}, and a {object_labels[-1]}")
    
    # Add OCR text results
    if ocr_text:
        description_parts.append(f"Text detected: {ocr_text}")
    
    return ". ".join(description_parts) + "."

def detect_objects(image_data):
    """
    Detect objects using MobileNet-SSD, extract text via OCR, and generate scene description.
    Returns all detection data including objects, OCR text, and scene description.
    """
    # Load model and prepare image
    model, weights, device = _load_model()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Preprocess image for model
    preprocess = weights.transforms()
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Run object detection
    with torch.no_grad():
        predictions = model(img_tensor)
    
    pred = predictions[0]
    categories = weights.meta['categories']
    detected_objects = []
    
    # Extract detection results
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    confidence_threshold = 0.5
    
    # Filter and format detected objects
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            x_min, y_min, x_max, y_max = box.astype(int).tolist()
            label_name = categories[label]
            
            detected_objects.append({
                'label': label_name,
                'confidence': float(score),
                'box': [x_min, y_min, x_max, y_max]
            })
    
    # Run OCR to extract text
    ocr_reader = _load_ocr_reader()
    image_np = np.array(image)
    ocr_results = ocr_reader.readtext(image_np, detail=0)
    ocr_text = ' '.join(ocr_results) if ocr_results else ''
    
    # Generate scene description
    scene_description = _generate_scene_description(detected_objects, ocr_text)
    
    # Build response with all data
    response = {
        'objects': detected_objects,
        'ocr_text': ocr_text,
        'scene_description': scene_description
    }
    
    return response
