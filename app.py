from flask import Flask, jsonify, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import gc
import json
from werkzeug.utils import secure_filename  # --- EXISTING IMPORT KEPT ---

from assistance_helpers import analyze_accident_scene, get_location, get_travel_assistance

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATHS = {
    "vgg16": "saved_models/accident_classifier_vgg16.keras",
    "vgg19": "saved_models/accident_classifier_vgg19.keras",
    "resnet50": "saved_models/accident_classifier_resnet50.keras",
    "googlenet": "saved_models/accident_classifier_inceptionv3.keras",
    "mobilenetv2": "saved_models/accident_classifier_original_mobilenet_cnn.keras",
    "vit": "saved_models/accident_classifier_vit.keras",
}

# --- ADDED: cache loaded models so prediction does not reload them every request ---
MODEL_CACHE = {}
MODEL_DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "resnet50": "ResNet50",
    "googlenet": "GoogleNet (InceptionV3)",
    "mobilenetv2": "MobileNetV2",
    "vit": "Vision Transformer (ViT)",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --- ADDED: custom layers required to load the saved ViT model ---
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_patches": self.num_patches, "projection_dim": self.projection_dim}
        )
        return config


# --- ADDED: separate ViT loader as requested ---
def load_vit_model():
    vit_path = MODEL_PATHS["vit"]
    if os.path.exists(vit_path):
        return load_model(
            vit_path,
            custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder},
        )
    return None


def get_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if model_name == "vit":
        model = load_vit_model()
        if model is not None:
            MODEL_CACHE[model_name] = model
        return model

    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(path):
        model = load_model(path)
        MODEL_CACHE[model_name] = model
        return model
    return None


def get_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            raw_metrics = json.load(f)
            mapped_metrics = {
                "mobilenetv2": raw_metrics.get("Original_MobileNet_CNN", {}),
                "resnet50": raw_metrics.get("ResNet50", {}),
                "googlenet": raw_metrics.get("InceptionV3", {}),
                "vgg16": raw_metrics.get("VGG16", {}),
                "vgg19": raw_metrics.get("VGG19", {}),
                "vit": raw_metrics.get("ViT", {}),
            }
            return mapped_metrics
    return {}


# --- ADDED: visual localization helpers for image/video accident regions ---
def is_image_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in IMAGE_EXTENSIONS


def get_target_size(model_name, model=None):
    if model is not None:
        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, list) and input_shape:
            input_shape = input_shape[0]

        if input_shape and len(input_shape) >= 3:
            height = input_shape[1]
            width = input_shape[2]
            if isinstance(height, int) and isinstance(width, int):
                return (width, height)

    return (224, 224) if model_name == "vit" else (250, 250)


def prepare_input_frame(frame, model_name, model=None):
    frame_resized = cv2.resize(frame, get_target_size(model_name, model))
    frame_normalized = frame_resized.astype("float32") / 255.0
    return np.expand_dims(frame_normalized, axis=0)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            layer_output = layer.output
        except (AttributeError, ValueError):
            continue

        output_shape = getattr(layer_output, "shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer
    return None


def make_gradcam_heatmap(model, image_tensor, last_conv_layer):
    predictions = model(image_tensor, training=False)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]],
    )

    with tf.GradientTape() as tape:
        conv_outputs, grad_predictions = grad_model(image_tensor, training=False)
        if grad_predictions.shape[-1] == 1:
            loss = grad_predictions[:, 0]
        else:
            predicted_index = tf.argmax(predictions[0])
            loss = grad_predictions[:, predicted_index]

    gradients = tape.gradient(loss, conv_outputs)
    if gradients is None:
        raise ValueError("Gradients could not be computed for the selected model.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_localized_overlay(frame, heatmap, output_path):
    heatmap_uint8 = np.uint8(255 * heatmap)
    resized_heatmap = cv2.resize(heatmap_uint8, (frame.shape[1], frame.shape[0]))
    colored_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)


def get_center_bbox(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    box_width = max(frame_width // 3, 1)
    box_height = max(frame_height // 3, 1)
    x = max((frame_width - box_width) // 2, 0)
    y = max((frame_height - box_height) // 2, 0)
    return {"x": x, "y": y, "width": box_width, "height": box_height}


def save_estimated_region_preview(frame, bbox, output_path):
    preview = frame.copy()
    start_point = (bbox["x"], bbox["y"])
    end_point = (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"])
    cv2.rectangle(preview, start_point, end_point, (0, 255, 255), 3)
    cv2.putText(
        preview,
        "Estimated accident region",
        (bbox["x"], max(bbox["y"] - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(output_path, preview)


def save_analysis_frame(frame, base_filename):
    analysis_filename = f"analysis_{os.path.splitext(base_filename)[0]}.jpg"
    analysis_path = os.path.join(UPLOAD_FOLDER, analysis_filename)
    cv2.imwrite(analysis_path, frame)
    return analysis_path


def extract_heatmap_bbox(heatmap, frame_shape):
    heatmap_uint8 = np.uint8(255 * heatmap)
    resized_heatmap = cv2.resize(heatmap_uint8, (frame_shape[1], frame_shape[0]))
    _, thresholded = cv2.threshold(resized_heatmap, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def describe_bbox_position(bbox, frame_shape):
    if not bbox:
        return "image center"

    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    width = frame_shape[1]
    height = frame_shape[0]

    horizontal = "left" if center_x < width / 3 else "right" if center_x > (2 * width / 3) else "center"
    vertical = "top" if center_y < height / 3 else "bottom" if center_y > (2 * height / 3) else "middle"

    if horizontal == "center" and vertical == "middle":
        return "image center"
    if horizontal == "center":
        return f"{vertical}-center"
    if vertical == "middle":
        return f"center-{horizontal}"
    return f"{vertical}-{horizontal}"


def build_accident_region_details(localization_hint, scene_analysis, model_name):
    if not localization_hint and not scene_analysis:
        return None

    region_position = None
    if scene_analysis:
        region_position = scene_analysis.get("accident_position_in_image")
    if not region_position and localization_hint:
        region_position = localization_hint.get("position_label")

    bbox = localization_hint.get("bbox") if localization_hint else None
    coordinates_text = None
    if bbox:
        coordinates_text = (
            f"x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, height={bbox['height']}"
        )

    preview_available = model_name != "vit"
    note = (
        "Approximate region estimated from the analyzed frame."
        if preview_available
        else "ViT model gives an approximate accident region text summary; heatmap preview is not available."
    )

    return {
        "position": region_position or "image center",
        "coordinates": coordinates_text,
        "note": note,
    }


def build_fallback_localization(frame, model_name, base_filename, message):
    bbox = get_center_bbox(frame.shape)
    localization_filename = f"localized_{os.path.splitext(base_filename)[0]}.jpg"
    localization_path = os.path.join(UPLOAD_FOLDER, localization_filename)
    save_estimated_region_preview(frame, bbox, localization_path)
    localization_hint = {
        "position_label": describe_bbox_position(bbox, frame.shape),
        "bbox": bbox,
    }
    fallback_note = (
        f"{message} Showing an estimated accident region instead."
        if model_name != "vit"
        else "ViT model uses an estimated region preview instead of a Grad-CAM heatmap."
    )
    return localization_path, fallback_note, localization_hint


def generate_localization_artifact(model, frame, model_name, base_filename):
    if model_name == "vit":
        return build_fallback_localization(
            frame,
            model_name,
            base_filename,
            "Localization preview is currently available for CNN-based models only.",
        )

    last_conv_layer = find_last_conv_layer(model)
    if not last_conv_layer:
        return build_fallback_localization(
            frame,
            model_name,
            base_filename,
            "Localization preview is not available for the selected model.",
        )

    try:
        image_tensor = prepare_input_frame(frame, model_name, model)
        heatmap = make_gradcam_heatmap(model, image_tensor, last_conv_layer)
    except Exception as e:
        print(f"Localization generation failed: {e}")
        return build_fallback_localization(
            frame,
            model_name,
            base_filename,
            "Localization preview could not be generated for the selected model.",
        )

    localization_filename = f"localized_{os.path.splitext(base_filename)[0]}.jpg"
    localization_path = os.path.join(UPLOAD_FOLDER, localization_filename)
    save_localized_overlay(frame, heatmap, localization_path)
    bbox = extract_heatmap_bbox(heatmap, frame.shape)
    localization_hint = {
        "position_label": describe_bbox_position(bbox, frame.shape),
        "bbox": bbox,
    }

    return localization_path, None, localization_hint


@app.route("/")
def home():
    metrics = get_metrics()
    return render_template(
        "index.html",
        metrics=metrics,
        selected_model_name=None,
        selected_model_display_name=None,
    )


# --- ADDED: standalone API route for travel assistance ---
@app.route("/get_assistance", methods=["POST"])
def get_assistance():
    payload = request.get_json(silent=True) or {}
    accident_detected = payload.get("accident_detected", False)
    location = payload.get("location")

    if not accident_detected:
        return jsonify({"error": "Travel assistance is only available when an accident is detected."}), 400

    if not location:
        location = get_location()

    assistance = get_travel_assistance(location)
    return jsonify({"location": location, "assistance": assistance})


@app.route("/predict", methods=["POST"])
def predict():
    metrics = get_metrics()
    location_details = None
    assistance_text = None
    scene_analysis = None
    accident_region_details = None
    localized_media_path = None
    localization_note = None
    localization_hint = None
    selected_model_name = None

    file = request.files.get("video")
    selected_model_name = request.form.get("model_choice")

    if not file or not file.filename or not selected_model_name:
        return render_template(
            "index.html",
            prediction="Error: Please upload a file and select a model.",
            color="danger",
            metrics=metrics,
            location_details=location_details,
            assistance_text=assistance_text,
            scene_analysis=scene_analysis,
            accident_region_details=accident_region_details,
            localized_media_path=localized_media_path,
            localization_note=localization_note,
            selected_model_name=selected_model_name,
            selected_model_display_name=None,
        )

    # --- SECURE THE FILENAME TO PREVENT OPENCV CRASHES ---
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    model_display_name = MODEL_DISPLAY_NAMES.get(selected_model_name, selected_model_name.upper())
    print(f"\nLoading Model: {model_display_name}...")
    model = get_model(selected_model_name)

    if model is None:
        return render_template(
            "index.html",
            prediction="Error: Model file not found on server.",
            color="danger",
            metrics=metrics,
            location_details=location_details,
            assistance_text=assistance_text,
            scene_analysis=scene_analysis,
            accident_region_details=accident_region_details,
            localized_media_path=localized_media_path,
            localization_note=localization_note,
            selected_model_name=selected_model_name,
            selected_model_display_name=model_display_name,
        )

    accident_detected = False
    detected_frame = None
    preview_frame = None

    # --- WRAP IN TRY/EXCEPT SO A BAD IMAGE DOESNT KILL FLASK ---
    try:
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        frames_to_skip = 15

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            frame_count += 1

            if frame_count == 1 or frame_count % frames_to_skip == 0:
                if preview_frame is None:
                    preview_frame = frame.copy()

                frame_expanded = prepare_input_frame(frame, selected_model_name, model)

                pred = model.predict(frame_expanded, verbose=0)[0][0]

                if pred > 0.5:
                    accident_detected = True
                    detected_frame = frame.copy()
                    break

        cap.release()
    except Exception as e:
        print(f"Error processing video/image: {e}")
        return render_template(
            "index.html",
            prediction="Error: Could not process that file format.",
            color="danger",
            metrics=metrics,
            location_details=location_details,
            assistance_text=assistance_text,
            scene_analysis=scene_analysis,
            accident_region_details=accident_region_details,
            localized_media_path=localized_media_path,
            localization_note=localization_note,
            selected_model_name=selected_model_name,
            selected_model_display_name=model_display_name,
        )
    finally:
        # --- ADDED: keep cached models loaded, only light cleanup after inference ---
        gc.collect()

    if accident_detected:
        result = f"Accident Detected (Using {model_display_name})"
        color = "danger"

        # --- ADDED: enrich result with accident-only location + travel assistance ---
        location_details = get_location()
        assistance_text = get_travel_assistance(location_details)
    else:
        result = f"No Accident Detected (Using {model_display_name})"
        color = "safe"

    localization_source_frame = detected_frame if detected_frame is not None else preview_frame
    if localization_source_frame is not None:
        localized_media_path, localization_note, localization_hint = generate_localization_artifact(
            model,
            localization_source_frame,
            selected_model_name,
            filename,
        )

    if accident_detected:
        analysis_image_path = (
            filepath
            if is_image_file(filename)
            else save_analysis_frame(localization_source_frame, filename)
            if localization_source_frame is not None
            else filepath
        )
        scene_analysis = analyze_accident_scene(analysis_image_path, localization_hint)
        accident_region_details = build_accident_region_details(
            localization_hint,
            scene_analysis,
            selected_model_name,
        )

    return render_template(
        "index.html",
        prediction=result,
        color=color,
        metrics=metrics,
        location_details=location_details,
        assistance_text=assistance_text,
        scene_analysis=scene_analysis,
        accident_region_details=accident_region_details,
        localized_media_path=localized_media_path,
        localization_note=localization_note,
        selected_model_name=selected_model_name,
        selected_model_display_name=model_display_name,
    )


if __name__ == "__main__":
    app.run(debug=False, threaded=False, use_reloader=False)
