print("--- ✅ RUNNING ENSEMBLE SCRIPT (ResNet, EfficientNet-B2, ConvNeXtTiny) ---")

import pandas as pd
import numpy as np
import os
from PIL import Image
import math
import tensorflow as tf
import tensorflow_addons as tfa # Needed for custom objects like AdamW and RandomCutout

from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION ---
# This section defines the models to be ensembled, including their file paths,
# required image sizes, and specific preprocessing functions.
MODELS_TO_ENSEMBLE = {
    'resnet_finetuned': {
        'path': 'resnet_model_finetuned.keras',
        'img_size': 224,
        'preprocess_func': tf.keras.applications.resnet.preprocess_input,
        'custom_objects': {
            'f1_micro': metrics.F1Score(average='micro', name="f1_micro")
        }
    },
    'efficientnet_b2_finetuned': {
        'path': 'b2_model_finetuned.keras',
        'img_size': 260,
        'preprocess_func': tf.keras.applications.efficientnet.preprocess_input,
        'custom_objects': {
            'f1_micro': metrics.F1Score(average='micro', name="f1_micro"),
            'AdamW': tfa.optimizers.AdamW # Custom optimizer from the training script
        }
    },
}

TTA_STEPS = 5
BATCH_SIZE = 16
SUBMISSION_FILE_PATH = '../dataset/sample_submission.csv'
TRAIN_CSV_PATH = '../dataset/train.csv'
TEST_IMAGE_DIR = '../dataset/test'
OUTPUT_FILENAME = 'submission_ensemble_resnet_b2_convnext.csv'

# --- 2. LOAD MODELS ---
print("\n--- Loading Ensemble Models ---")
models = {}
for name, model_info in MODELS_TO_ENSEMBLE.items():
    try:
        print(f"-> Loading '{name}' from {model_info['path']}...")
        models[name] = load_model(model_info['path'], custom_objects=model_info['custom_objects'])
    except Exception as e:
        print(f"❌ Error loading model '{name}': {e}")
        print("   Please ensure the model file exists and all necessary custom objects are defined.")
        exit()
print("✅ All models loaded successfully.")


# --- 3. SETUP DATA AND LABEL ENCODER ---
try:
    df_test = pd.read_csv(SUBMISSION_FILE_PATH)
    df_train_for_fit = pd.read_csv(TRAIN_CSV_PATH)
except FileNotFoundError as e:
    print(f"❌ Error: CSV file not found. {e}")
    exit()

print("\n⏳ Setting up Label Encoder...")
le = LabelEncoder()
le.fit(df_train_for_fit['TARGET'])
print("✅ Label Encoder setup complete.")


# --- 4. TTA AND ENSEMBLE PREDICTION ---
tta_datagen = ImageDataGenerator(
    horizontal_flip=True, rotation_range=15, zoom_range=0.15
)
all_final_predictions = []
num_batches = math.ceil(len(df_test) / BATCH_SIZE)

print(f"\n🚀 Starting Ensemble TTA prediction on {len(df_test)} images...")

for i in range(num_batches):
    start_index = i * BATCH_SIZE
    end_index = min((i + 1) * BATCH_SIZE, len(df_test))
    batch_df = df_test.iloc[start_index:end_index]
    
    # Store predictions for each model for the current batch
    model_predictions_for_batch = []
    
    print(f"   -> Processing Batch {i+1}/{num_batches}...")
    
    # Iterate through each model to get its prediction for the batch
    for name, model in models.items():
        model_info = MODELS_TO_ENSEMBLE[name]
        IMG_SIZE = model_info['img_size']
        preprocess_func = model_info['preprocess_func']
        
        # Prepare image data for the current model's requirements
        batch_data = []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(TEST_IMAGE_DIR, row['ID'])
            img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            arr_img = preprocess_func(np.array(img))
            batch_data.append(arr_img)
        batch_data = np.array(batch_data)
        
        # Perform TTA
        tta_preds = []
        tta_preds.append(model.predict(batch_data, verbose=0)) # Original image
        for _ in range(TTA_STEPS - 1): # Augmented images
            augmented_batch = next(tta_datagen.flow(batch_data, batch_size=len(batch_data), shuffle=False))
            tta_preds.append(model.predict(augmented_batch, verbose=0))
            
        # Average the TTA predictions for the current model
        avg_tta_preds = np.mean(tta_preds, axis=0)
        model_predictions_for_batch.append(avg_tta_preds)

    # Average the predictions from all models (ensemble)
    final_ensembled_preds = np.mean(model_predictions_for_batch, axis=0)
    all_final_predictions.append(final_ensembled_preds)

print("\n✅ Ensemble TTA Predictions for all batches generated.")

# --- 5. CREATE SUBMISSION FILE ---
print("\n--- Creating Submission File ---")
# Combine predictions from all batches
final_predictions = np.vstack(all_final_predictions)

# Get the class with the highest probability
predicted_labels_encoded = np.argmax(final_predictions, axis=1)

# Convert encoded labels back to original class names
predicted_labels = le.inverse_transform(predicted_labels_encoded)

submission_df = pd.DataFrame({'ID': df_test['ID'], 'TARGET': predicted_labels})
submission_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Final submission file '{OUTPUT_FILENAME}' created successfully!")
print("\nHere's a preview of your submission file:")
print(submission_df.head())