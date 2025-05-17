from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from telegram import Bot
import asyncio

app = Flask(__name__)

# Load YOLOv8 pretrained model
model = YOLO('yolov8x.pt')

# Load ONNX model for ambulance detection
onnx_model_path = 'model.onnx'  # Path to your ONNX model
ambulance_model = ort.InferenceSession(onnx_model_path)

# Vehicle weights for green time calculation
vehicle_weights = {'car': 5, 'motorcycle': 3, 'bus': 7, 'truck': 10, 'van': 10}

# Upload folder for junction images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Telegram Bot Token and Chat ID
TELEGRAM_BOT_TOKEN = '7755373107:AAHGILKELCKNQZR5Iggsv_HVIEHxuPRlMl0'
TELEGRAM_CHAT_ID = '7565137984'

# Initialize Telegram Bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_telegram_alert(image_path, junction_id):
    message = f"🚨 Ambulance detected at Junction {junction_id}! Please take immediate action."
    with open(image_path, 'rb') as photo:
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=message)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Save uploaded images
    junctions = {}
    for i in range(1, 5):
        file = request.files[f'junction{i}']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'junction{i}.jpg')
        file.save(path)
        junctions[f'junction{i}'] = path
    return render_template('simulation.html', junctions=junctions)

def preprocess_image(img):
    """
    Preprocess the image for ONNX model inference.
    """
    # Convert BGR to RGB and resize with PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((640, 640))  # Resize once with PIL
    
    # Convert to numpy array (uint8) and add batch dimension
    input_data = np.array(img_pil).astype(np.uint8)  # Use uint8
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    return input_data

def detect_ambulance(img):
    """
    Detect ambulance in the image using the ONNX model.
    """
    try:
        # Preprocess the image
        input_data = preprocess_image(img)

        # Run inference
        input_name = ambulance_model.get_inputs()[0].name
        outputs = ambulance_model.run(None, {input_name: input_data})

        # Extract detection results
        detection_boxes = outputs[0][0]  # Shape: [N, 4]
        detection_scores = outputs[1][0]  # Shape: [N]
        detection_classes = outputs[2][0]  # Shape: [N]
        num_detections = int(outputs[3][0])  # Shape: [1]

        # Check for ambulance detections (class 1) with confidence > 0.5
        ambulance_detected = False
        for i in range(num_detections):
            if detection_classes[i] == 1 and detection_scores[i] > 0.5:  # Class 1 is ambulance
                ambulance_detected = True
                break

        print(f"Ambulance Detected: {ambulance_detected}")
        return ambulance_detected
    except Exception as e:
        print(f"Ambulance detection error: {str(e)}")
        return False

def process_junction(junction_id):
    """
    Process a single junction's image.
    Returns: (green_time, detected_vehicles, ambulance_detected)
    """
    try:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'junction{junction_id}.jpg')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image for junction {junction_id} not found")

        # Read image (DO NOT RESIZE HERE)
        img = cv2.imread(path)

        # Vehicle detection with YOLO (resize here if needed for YOLO)
        img_for_yolo = cv2.resize(img, (640, 640))  # YOLO expects 640x640
        results = model.predict(img_for_yolo, conf=0.5)
        total_time = 0
        vehicles = set()
        ambulance_detected = False

        for box in results[0].boxes:
            cls = model.names[int(box.cls)]
            confidence = box.conf.item()
            
            # Check if YOLO detects a van or truck
            if cls in ['van', 'truck'] and confidence >= 0.5:
                # Run ONNX model to check for ambulance
                ambulance_detected = detect_ambulance(img)
                if ambulance_detected:
                    cls = "ambulance"  # Override the class to ambulance
                else:
                    cls = cls  # Keep the original class

            # Add to total time and vehicles set
            if cls in vehicle_weights and confidence >= 0.5:
                total_time += vehicle_weights[cls]
                vehicles.add(cls)

        # If ambulance is detected, send an alert to Telegram
        if ambulance_detected:
            asyncio.run(send_telegram_alert(path, junction_id))

        return (
            max(0, min(total_time, 60)),
            ", ".join(vehicles) or "None",
            ambulance_detected
        )
    except Exception as e:
        print(f"Junction {junction_id} error: {str(e)}")
        return 0, "Error", False

@app.route('/simulate')
def simulate():
    results = {
        'green_times': {},
        'detected_vehicles': {},
        'ambulance_flags': {}
    }

    for i in range(1, 5):
        green_time, vehicles, ambulance = process_junction(i)
        results['green_times'][f'junction{i}'] = green_time
        results['detected_vehicles'][f'junction{i}'] = vehicles
        results['ambulance_flags'][f'junction{i}'] = ambulance

    # Debug: Print the results
    print("Simulation Results:", results)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False, port=5000)