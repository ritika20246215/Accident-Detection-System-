import os
import gc
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D

# --- 1. OPTIMIZATION: Prevent GPU Memory Crashes ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# --- 2. SETUP ---
IMG_SIZE = 224
BATCH_SIZE = 32 # Optimal training speed

# ================= DATA SETUP =================
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=20,      
    zoom_range=0.2,         
    horizontal_flip=True    
)

val_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    r'C:\Users\pushk\OneDrive\Desktop\finalads\Accident Images Analysis Dataset\train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    r'C:\Users\pushk\OneDrive\Desktop\finalads\Accident Images Analysis Dataset\train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- DATASET CHECK ---
print("\n" + "="*40)
print("📊 DATASET VERIFICATION")
print(f"Classes Found: {train_data.class_indices}")
print(f"Total Training Images: {train_data.samples}")
print(f"Total Validation Images: {val_data.samples}")
print("="*40 + "\n")

def clean_up():
    """Clears RAM/VRAM to prevent crashes between model training."""
    tf.keras.backend.clear_session()
    gc.collect()
    print("-" * 40)

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# ================= METRICS SETUP =================
metrics_data = {}

def get_eval_metrics():
    """Returns a fresh list of Keras metrics for compilation to prevent state overlap."""
    return [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

def save_metrics(history, model_key):
    """Safely extracts validation metrics, calculates F1, and saves to JSON."""
    hist = history.history
    
    # Fuzzy search for the exact dictionary keys Keras generated for this run
    acc_key = next((k for k in hist.keys() if k.startswith('val_') and 'acc' in k), None)
    prec_key = next((k for k in hist.keys() if k.startswith('val_') and 'prec' in k), None)
    rec_key = next((k for k in hist.keys() if k.startswith('val_') and 'rec' in k), None)

    # Extract the final epoch's numbers safely
    val_acc = hist[acc_key][-1] * 100 if acc_key else 0.0
    val_prec = hist[prec_key][-1] * 100 if prec_key else 0.0
    val_rec = hist[rec_key][-1] * 100 if rec_key else 0.0

    # Calculate F1-Score
    if (val_prec + val_rec) == 0:
        val_f1 = 0.0
    else:
        val_f1 = 2 * (val_prec * val_rec) / (val_prec + val_rec)

    # Store in dictionary
    metrics_data[model_key] = {
        "accuracy": round(val_acc, 2),
        "precision": round(val_prec, 2),
        "recall": round(val_rec, 2),
        "f1_score": round(val_f1, 2)
    }

    # Save to JSON file dynamically after each model completes
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"📊 Metrics saved for {model_key.upper()}: Acc={val_acc:.1f}%, Prec={val_prec:.1f}%, Rec={val_rec:.1f}%, F1={val_f1:.1f}%")


# ================= 1. CNN MODEL =================
print("\n[1/6] Training Custom CNN...")
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=get_eval_metrics())
# CHANGED: epochs=2
history_cnn = cnn_model.fit(train_data, epochs=20, validation_data=val_data)
save_metrics(history_cnn, "cnn")
cnn_model.save("saved_models/cnn_model.keras") 
clean_up()


# ================= 2. VGG16 =================
print("\n[2/6] Training VGG16...")
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
vgg16_base.trainable = False 

vgg16_model = Sequential([
    vgg16_base,
    GlobalAveragePooling2D(), 
    Dense(128, activation='relu'),
    Dropout(0.5),             
    Dense(1, activation='sigmoid')
])

vgg16_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=get_eval_metrics())
# CHANGED: epochs=2
history_vgg16 = vgg16_model.fit(train_data, epochs=2, validation_data=val_data)
save_metrics(history_vgg16, "vgg16")
vgg16_model.save("saved_models/vgg16_model.keras")
clean_up()


# ================= 3. VGG19 =================
print("\n[3/6] Training VGG19...")
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
vgg19_base.trainable = False

vgg19_model = Sequential([
    vgg19_base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),             
    Dense(1, activation='sigmoid')
])

vgg19_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=get_eval_metrics())
# CHANGED: epochs=2
history_vgg19 = vgg19_model.fit(train_data, epochs=2, validation_data=val_data)
save_metrics(history_vgg19, "vgg19")
vgg19_model.save("saved_models/vgg19_model.keras")
clean_up()


# ================= 4. RESNET50 =================
print("\n[4/6] Training ResNet50...")
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
resnet_base.trainable = False

resnet_model = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=get_eval_metrics())
# CHANGED: epochs=2
history_resnet = resnet_model.fit(train_data, epochs=2, validation_data=val_data)
save_metrics(history_resnet, "resnet50")
resnet_model.save("saved_models/resnet50_model.keras")
clean_up()


# ================= 5. GOOGLENET (InceptionV3) =================
print("\n[5/6] Training GoogleNet (InceptionV3)...")
inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
inception_base.trainable = False

inception_model = Sequential([
    inception_base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

inception_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=get_eval_metrics())
# CHANGED: epochs=2
history_googlenet = inception_model.fit(train_data, epochs=2, validation_data=val_data)
save_metrics(history_googlenet, "googlenet")
inception_model.save("saved_models/googlenet_model.keras")
clean_up()


# ================= 6. MOBILENET V2 =================
print("\n[6/6] Training MobileNetV2...")
mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
mobilenet_base.trainable = False

mobilenet_model = Sequential([
    mobilenet_base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

mobilenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=get_eval_metrics())
# CHANGED: epochs=2
history_mobilenet = mobilenet_model.fit(train_data, epochs=2, validation_data=val_data)
save_metrics(history_mobilenet, "mobilenetv2")
mobilenet_model.save("saved_models/mobilenetv2_model.keras")
clean_up()

print("\n🚀 ALL 6 MODELS SUCCESSFULLY TRAINED AND METRICS SAVED!")