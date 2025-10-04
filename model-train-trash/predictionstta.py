import pandas as pd
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import metrics
import math
import tensorflow_addons as tfa

# --- Custom object needed to load the models ---
custom_objects = {
    'f1_micro': metrics.F1Score(average='micro', name="f1_micro"),
    'AdamW': tfa.optimizers.AdamW
}

# --- Model and Path Configuration ---
MODEL_CONFIG = {
    'efficientnet_b1': {
        'path': 'best_model_finetuned.keras',
        'img_size': 240
    },
    'efficientnet_b2': {
        'path': 'b2_model_finetuned.keras',
        'img_size': 260
    }
}
# --- Load Label Encoder and Test File List ---
try:
    df_test = pd.read_csv("../dataset/sample_submission.csv")
    df_train_for_fit = pd.read_csv("../dataset/train.csv")
except FileNotFoundError as e:
    print(f"❌ Error: CSV file not found. {e}")
    exit()

le = LabelEncoder()
le.fit(df_train_for_fit['TARGET'])
test_image_dir = "../dataset/test"

all_model_predictions = []

# --- Loop Through Each Model to Get Predictions ---
for model_name, config in MODEL_CONFIG.items():
    print(f"\n--- Loading and Predicting with {model_name} ---")
    try:
        model = load_model(config['path'], custom_objects=custom_objects)
        print(f"✅ Model '{config['path']}' loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model {config['path']}. Please ensure it's named correctly. Error: {e}")
        continue

    IMG_SIZE = config['img_size']
    tta_steps = 8
    batch_size = 32
    all_avg_predictions = []
    num_batches = math.ceil(len(df_test) / batch_size)
    
    tta_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, zoom_range=0.15)

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(df_test))
        batch_df = df_test.iloc[start_index:end_index]
        
        batch_data = []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(test_image_dir, row['ID'])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                arr_img = preprocess_input(np.array(img))
                batch_data.append(arr_img)
        
        batch_data = np.array(batch_data)
        if batch_data.size == 0: continue

        batch_predictions_tta = []
        print(f"  -> {model_name}: Processing Batch {i+1}/{num_batches}")
        for _ in range(tta_steps):
            preds = model.predict(tta_datagen.flow(batch_data, batch_size=batch_size, shuffle=False), verbose=0)
            batch_predictions_tta.append(preds)
            
        avg_batch_preds = np.mean(batch_predictions_tta, axis=0)
        all_avg_predictions.append(avg_batch_preds)

    final_predictions_for_model = np.vstack(all_avg_predictions)
    all_model_predictions.append(final_predictions_for_model)


# --- ⭐ CHANGE: Ensemble with Weighted Averaging ---
if len(all_model_predictions) < 2:
    print("\n❌ Could not generate predictions from both models. Cannot create ensemble. Exiting.")
    exit()

print("\n🚀 Performing WEIGHTED averaging on predictions...")
# Give 30% weight to the B1 model and 70% to the stronger B2 model
b1_preds = all_model_predictions[0]
b2_preds = all_model_predictions[1]
ensemble_predictions = (0.3 * b1_preds) + (0.7 * b2_preds)
print("✅ Ensembling complete.")

# --- Create Final Submission File ---
predicted_labels_encoded = np.argmax(ensemble_predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_labels_encoded)

submission_df = pd.DataFrame({'ID': df_test['ID'], 'TARGET': predicted_labels})
submission_df.to_csv('submission_ensemble_weighted.csv', index=False)

print("\n✅ Final weighted ensemble submission file 'submission_ensemble_weighted.csv' created successfully!")
print("\nHere's a preview of your submission file:")
print(submission_df.head())