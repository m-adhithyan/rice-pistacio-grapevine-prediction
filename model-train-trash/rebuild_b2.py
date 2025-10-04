import tensorflow as tf
from tensorflow.keras import layers, models

# Configuration to match the original B2 model
IMG_SIZE = 260
NUM_CLASSES = 20
WEIGHTS_PATH = 'b2_weights_only.h5'
NEW_MODEL_PATH = 'b2_model_rebuilt.keras'

print("--- Rebuilding EfficientNet-B2 model from .h5 weights ---")

# 1. Build the B2 architecture
base_model = tf.keras.applications.EfficientNetB2(
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
print("✅ B2 model structure created successfully.")

# 2. Load weights from the .h5 file
try:
    print(f"-> Loading weights from '{WEIGHTS_PATH}'...")
    model.load_weights(WEIGHTS_PATH)
    print("✅ B2 weights loaded successfully.")
except Exception as e:
    print(f"❌ Error loading B2 weights: {e}")
    exit()

# 3. Save the new, clean model
model.save(NEW_MODEL_PATH)
print(f"\n✅ Successfully rebuilt and saved the B2 model to '{NEW_MODEL_PATH}'")