import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S

print("--- Building model architecture in the modern environment ---")
IMG_SIZE = 384
NUM_CLASSES = 20 # Make sure this matches your dataset

# Recreate the exact model architecture
base_model = EfficientNetV2S(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

print("--- Loading weights from the old, incompatible model file ---")
# Load only the weights into the new architecture
model.load_weights('efficientnetv2s_model.keras')

print("--- Saving a new, fully compatible model file ---")
# Save the entire model in the new, modern format
model.save('efficientnetv2s_model_v2.keras')
print("✅ New model 'efficientnetv2s_model_v2.keras' saved.")