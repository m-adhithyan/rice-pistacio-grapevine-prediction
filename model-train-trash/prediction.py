import pandas as pd
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
# --- CHANGE: Imported the correct preprocessing function for EfficientNetV2 ---
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import metrics
import math

# --- Instructions ---
# 1. Save this code as a Python file (e.g., predict_efficientnetv2s.py).
# 2. Place it in the same folder where your 'netv2s_finetuned.keras' is located.
# 3. Ensure your dataset folder is structured correctly relative to this script.
# 4. Run from your terminal: python predict_efficientnetv2s.py

# --- Custom object dictionary needed to load the model with the F1Score metric ---
custom_objects = {
    'f1_micro': metrics.F1Score(average='micro', name="f1_micro")
}

# --- 1. Load your BEST fine-tuned model ---
# --- CHANGE: Using the EfficientNetV2S model name from your training output ---
try:
    model = load_model('netv2s_finetuned.keras', custom_objects=custom_objects)
    print("✅ Model 'netv2s_finetuned.keras' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 2. Setup file paths and Label Encoder ---
try:
    df_test = pd.read_csv("../dataset/sample_submission.csv")
    df_train_for_fit = pd.read_csv("../dataset/train.csv")
except FileNotFoundError as e:
    print(f"❌ Error: CSV file not found. {e}")
    exit()

test_image_dir = "../dataset/test"
# --- CHANGE: Image size must match the 384x384 used to train the EfficientNetV2S model ---
IMG_SIZE = 384

print("⏳ Setting up Label Encoder...")
le = LabelEncoder()
le.fit(df_train_for_fit['TARGET'])
print("✅ Label Encoder setup complete.")


# --- 3. Memory-Efficient TTA Prediction in Batches ---
# This ImageDataGenerator is used to create augmented versions of the test images.
tta_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15]
)

tta_steps = 5      # Number of augmented versions to create for each image
batch_size = 16    # Process 16 images at a time (good for larger 384x384 images)
all_avg_predictions = []
num_batches = math.ceil(len(df_test) / batch_size)

print(f"\n🚀 Starting TTA prediction on {len(df_test)} images in {num_batches} batches...")

for i in range(num_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(df_test))
    batch_df = df_test.iloc[start_index:end_index]

    batch_data = []
    for _, row in batch_df.iterrows():
        img_path = os.path.join(test_image_dir, row['ID'])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            # --- CHANGE: Preprocess the image for EfficientNetV2 ---
            arr_img = preprocess_input(np.array(img))
            batch_data.append(arr_img)

    batch_data = np.array(batch_data)

    if batch_data.size == 0:
        continue

    # Store all augmented predictions for the current batch
    tta_predictions_for_batch = []
    print(f"  -> Processing Batch {i+1}/{num_batches} (images {start_index+1}-{end_index})")

    # 1. Get the original prediction (no augmentation)
    original_preds = model.predict(batch_data, verbose=0)
    tta_predictions_for_batch.append(original_preds)
    
    # 2. Get predictions on augmented images
    for tta_step in range(tta_steps - 1): # We already did one step
        # Get one batch of augmented images from the generator
        augmented_batch = next(tta_datagen.flow(batch_data, batch_size=len(batch_data), shuffle=False))
        preds = model.predict(augmented_batch, verbose=0)
        tta_predictions_for_batch.append(preds)

    # Average all the TTA predictions for this batch
    avg_batch_preds = np.mean(tta_predictions_for_batch, axis=0)
    all_avg_predictions.append(avg_batch_preds)

print("\n✅ TTA Predictions for all batches generated and averaged.")

# --- 4. Combine results and create submission file ---
final_predictions = np.vstack(all_avg_predictions)

predicted_labels_encoded = np.argmax(final_predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_labels_encoded)

submission_df = pd.DataFrame({'ID': df_test['ID'], 'TARGET': predicted_labels})
# --- CHANGE: Made filename specific to the model ---
submission_filename = 'submission_efficientnetv2s.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\n✅ Final submission file '{submission_filename}' created successfully!")
print("\nHere's a preview of your submission file:")
print(submission_df.head())