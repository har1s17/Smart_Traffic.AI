import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Assuming you have YOLOv8x installed

# Load the ONNX model
onnx_model_path = 'model.onnx'  # Replace with your model path
ambulance_model = ort.InferenceSession(onnx_model_path)

# Load the YOLOv8x model
yolo_model = YOLO('yolov8x.pt')  # Replace with your YOLOv8x model path

# Load an image
image_path = '00bike.jpeg'  # Replace with your test image
img = cv2.imread(image_path)

# Preprocess the image
def preprocess_image(img):
    # Convert BGR to RGB and resize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((640, 640))
    
    # Convert to numpy array (uint8) and add batch dimension
    input_data = np.array(img_pil).astype(np.uint8)  # Use uint8 instead of float32
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, img  # Return original image for annotation

# Run inference & annotate image
def detect_and_annotate(img):
    try:
        # Preprocess the image
        input_data, original_img = preprocess_image(img)

        # Run YOLOv8x inference
        yolo_results = yolo_model(img)
        yolo_detections = yolo_results[0].boxes.data.cpu().numpy()

        # Run ONNX model inference
        input_name = ambulance_model.get_inputs()[0].name
        outputs = ambulance_model.run(None, {input_name: input_data})

        # Extract ONNX model detection results
        detection_boxes = outputs[0][0]  # Shape: [N, 4]
        detection_scores = outputs[1][0]  # Shape: [N]
        detection_classes = outputs[2][0]  # Shape: [N]
        num_detections = int(outputs[3][0])  # Shape: [1]

        height, width, _ = original_img.shape
        ambulance_detected = False

        # Check YOLOv8x detections
        for detection in yolo_detections:
            x1, y1, x2, y2, conf, cls = detection
            label = yolo_model.names[int(cls)]

            # If YOLO detects a van or truck, check ONNX model for ambulance
            if label in ['van', 'truck']:
                for i in range(num_detections):
                    if detection_classes[i] == 1 and detection_scores[i] > 0.8:  # Class 1 = Ambulance
                        ambulance_detected = True
                        # Extract bounding box coordinates (normalized)
                        ymin, xmin, ymax, xmax = detection_boxes[i]
                        left, right, top, bottom = (int(xmin * width), int(xmax * width),
                                                    int(ymin * height), int(ymax * height))

                        # Draw bounding box
                        cv2.rectangle(original_img, (left, top), (right, bottom), (0, 255, 0), 3)
                        label = f"Ambulance ({detection_scores[i]:.2f})"
                        cv2.putText(original_img, label, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break
            else:
                # Draw bounding box for other YOLO detections
                cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"{label} ({conf:.2f})"
                cv2.putText(original_img, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Ambulance Detected: {ambulance_detected}")

        # Save the annotated image
        output_path = "output.jpg"
        cv2.imwrite(output_path, original_img)
        print(f"Annotated image saved as {output_path}")

        return ambulance_detected, output_path
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, None

# Test with image
ambulance_detected, output_image_path = detect_and_annotate(img)

# Show the saved image (works for all environments)
if output_image_path:
    img_annotated = cv2.imread(output_image_path)

    # Convert BGR to RGB for correct colors in Matplotlib
    img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

    # Display using Matplotlib (for Colab, Jupyter, and GUI environments)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")  # Hide axis
    plt.title("Ambulance Detection")
    plt.show()

    print(f"Final Result: Ambulance Detected: {ambulance_detected}")