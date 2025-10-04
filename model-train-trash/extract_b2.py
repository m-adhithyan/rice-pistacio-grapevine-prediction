import tensorflow as tf

# We need to provide the custom objects to load the original model
custom_objects = {
    'f1_micro': tf.keras.metrics.F1Score(average='micro', name="f1_micro"),
    'AdamW': tf.keras.optimizers.AdamW
}

OLD_MODEL_PATH = 'b2_model_finetuned.keras'
WEIGHTS_PATH = 'b2_weights_only.h5'

print(f"--- Extracting weights from '{OLD_MODEL_PATH}' ---")

# Load the original model
try:
    model = tf.keras.models.load_model(OLD_MODEL_PATH, custom_objects=custom_objects)
    print("✅ Original B2 model loaded successfully.")
except Exception as e:
    print(f"❌ Could not load original model. Make sure the file '{OLD_MODEL_PATH}' exists.")
    print(f"Error: {e}")
    exit()

# Save ONLY the weights to a .h5 file
model.save_weights(WEIGHTS_PATH)

print(f"\n✅ B2 weights extracted and saved to '{WEIGHTS_PATH}'")