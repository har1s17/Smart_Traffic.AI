import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "frozen_inference_graph.pb")
OUTPUT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "detected_ambulance.jpg")

# Load the TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_PATH, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

def upload_file():
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an image file", 
        filetypes=[("All Files", "*.*"), ("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        print("❌ No file selected")
        return
    detect_ambulance(file_path)

def detect_ambulance(image_path):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Load image
            image = cv2.imread(image_path)

            # Check if the image is loaded properly
            if image is None:
                print("❌ Error: Could not load image. Please check the file format.")
                return
            else:
                h, w, _ = image.shape
                image_expanded = np.expand_dims(image, axis=0)

                # Get input & output tensors
                image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
                detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
                detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
                detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
                num_detections = detection_graph.get_tensor_by_name("num_detections:0")

                # Run object detection
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded},
                )

                # Process detections
                for i in range(int(num[0])):
                    confidence = scores[0][i]
                    if confidence > 0.8:  # Confidence threshold
                        y1, x1, y2, x2 = boxes[0][i]
                        start_point = (int(x1 * w), int(y1 * h))
                        end_point = (int(x2 * w), int(y2 * h))

                        # Draw bounding box
                        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

                        # Add label with confidence score
                        label = f"Ambulance: {confidence:.2f}"
                        cv2.putText(
                            image,
                            label,
                            (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Save output image
                cv2.imwrite(OUTPUT_IMAGE_PATH, image)
                print(f"✅ Output saved as {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    upload_file()
