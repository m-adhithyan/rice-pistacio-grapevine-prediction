import tensorflow as tf

OLD_MODEL_PATH = 'resnet_model_finetuned.keras'
WEIGHTS_PATH = 'resnet_weights_only.h5'

print(f"--- Extracting weights from '{OLD_MODEL_PATH}' ---")

# Load the original model without compiling
try:
    model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)
    print("✅ Original model loaded successfully.")
except Exception as e:
    print(f"❌ Could not load original model. Make sure you are in the tf_legacy environment and the file exists.")
    print(f"Error: {e}")
    exit()

# Save ONLY the weights to a .h5 file
model.save_weights(WEIGHTS_PATH)

print(f"\n✅ Weights have been extracted and saved to '{WEIGHTS_PATH}'")
print("\nNext, activate your main TensorFlow environment and run the rebuild script.")