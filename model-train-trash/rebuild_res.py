# This is the updated rebuild.py script
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
NUM_CLASSES = 20
WEIGHTS_PATH = 'resnet_weights_only.h5'  # <-- Loading from the new weights file
NEW_MODEL_PATH = 'resnet_model_rebuilt.keras'

print("--- Rebuilding ResNet model from .h5 weights ---")

# 1. Build the exact same model architecture
base_model = tf.keras.applications.ResNet50(
    weights=None,
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
print("✅ Model structure created successfully.")

# 2. Load the weights from the .h5 file
try:
    print(f"-> Loading weights from '{WEIGHTS_PATH}'...")
    model.load_weights(WEIGHTS_PATH)
    print("✅ Weights loaded successfully.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("Please ensure you have run extract_weights.py first.")
    exit()

# 3. Save the new, clean Keras v3 model
model.save(NEW_MODEL_PATH)
print(f"\n✅ Successfully rebuilt and saved the model to '{NEW_MODEL_PATH}'")