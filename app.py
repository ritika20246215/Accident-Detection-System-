from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import gc
import json

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary mapping frontend dropdown choices to our saved .keras files
MODEL_PATHS = {
    "cnn": "saved_models/cnn_model.keras",
    "vgg16": "saved_models/vgg16_model.keras",
    "vgg19": "saved_models/vgg19_model.keras",
    "resnet50": "saved_models/resnet50_model.keras",
    "googlenet": "saved_models/googlenet_model.keras",
    "mobilenetv2": "saved_models/mobilenetv2_model.keras"
}

def get_model(model_name):
    """Safely loads the requested model from disk."""
    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(path):
        return load_model(path)
    return None

def get_metrics():
    """Loads the model scores from the JSON file generated during training."""
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    return {}

@app.route("/")
def home():
    # Pass the metrics to the frontend as soon as the page loads
    metrics = get_metrics()
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    # Load metrics so the right-side dashboard doesn't disappear on reload
    metrics = get_metrics()
    
    # 1. Get the file and the model choice from the HTML form
    file = request.files.get("video")
    selected_model_name = request.form.get("model_choice") 

    # Safety check: Did the user forget to upload or select a model?
    if not file or not file.filename or not selected_model_name:
        return render_template("index.html", prediction="Error: Please upload a file and select a model.", color="danger", metrics=metrics)

    # 2. Save the uploaded file temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # 3. CRITICAL: Clear memory before loading a new model to prevent RAM crashes
    tf.keras.backend.clear_session()
    gc.collect()

    print(f"\nLoading Model: {selected_model_name.upper()}...")
    model = get_model(selected_model_name)

    if model is None:
        return render_template("index.html", prediction="Error: Model file not found on server.", color="danger", metrics=metrics)

    # 4. Open the video (or image) using OpenCV
    cap = cv2.VideoCapture(filepath)
    accident_detected = False
    
    frame_count = 0
    frames_to_skip = 15  # Only check 1 out of every 15 frames for videos

    # 5. Process the file
    while cap.isOpened():
        ret, frame = cap.read()
        
        # SAFETY CHECK: If the video is over, or the frame is corrupted, stop.
        if not ret or frame is None:
            break 
            
        frame_count += 1
        
        # We always process the 1st frame (crucial for images!)
        # For videos, we skip 14 frames, process the 15th, skip 14, process the 30th, etc.
        if frame_count == 1 or frame_count % frames_to_skip == 0:
            
            # Format the image perfectly for our models (224x224)
            frame_resized = cv2.resize(frame, (224, 224))
            frame_normalized = frame_resized / 255.0
            frame_expanded = np.expand_dims(frame_normalized, axis=0)

            # Make the prediction
            pred = model.predict(frame_expanded, verbose=0)[0][0]

            # If the model is more than 50% sure it's an accident, trigger the alert!
            if pred > 0.5:
                accident_detected = True
                break

    cap.release()

    # 6. CRITICAL: Delete the model from memory after we are done to prevent crashes
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    # 7. Send the results back to the website
    if accident_detected:
        result = f"🚨 Accident Detected (Using {selected_model_name.upper()})"
        color = "danger"
    else:
        result = f"✅ No Accident Detected (Using {selected_model_name.upper()})"
        color = "safe"

    return render_template("index.html", prediction=result, color=color, metrics=metrics)

if __name__ == "__main__":
    # threaded=False stops TensorFlow from panicking and crashing!
    app.run(debug=True, threaded=False)