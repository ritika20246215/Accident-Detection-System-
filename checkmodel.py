import os
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model

# The exact paths to the models we just saved
MODEL_PATHS = {
    "CNN": "saved_models/cnn_model.keras",
    "VGG16": "saved_models/vgg16_model.keras",
    "VGG19": "saved_models/vgg19_model.keras",
    "ResNet50": "saved_models/resnet50_model.keras",
    "GoogleNet": "saved_models/googlenet_model.keras",
    "MobileNetV2": "saved_models/mobilenetv2_model.keras"
}

print("🔍 Starting Model Health Check...\n")
print("-" * 40)

# Loop through each model in our dictionary
for model_name, path in MODEL_PATHS.items():
    # 1. Check if the file actually exists on your hard drive
    if os.path.exists(path):
        try:
            print(f"⏳ Attempting to load {model_name}...")
            
            # 2. Try to physically load the model into Keras
            model = load_model(path)
            print(f"✅ SUCCESS: {model_name} is perfectly healthy!")
            
            # 3. Wipe it from memory so the computer doesn't crash
            del model
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            # If the file is corrupted, it will print this error
            print(f"❌ ERROR: {model_name} failed to load. Details: {e}")
    else:
        # If the file is missing from the folder
        print(f"⚠️ WARNING: Cannot find the file for {model_name} at {path}")
        
    print("-" * 40)

print("\n🏁 Diagnostic Check Complete!")