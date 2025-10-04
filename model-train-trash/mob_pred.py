import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, metrics
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. Setup Paths and Constants ---
print("--- Step 1: Setting up all paths and constants ---")
TEST_IMAGE_DIR = "../dataset/test"
TRAIN_CSV_PATH = "../dataset/train.csv"
IMG_SIZE = 224

# Models to test
INITIAL_MODEL_PATH = 'mobilenetv2_model.keras'
FINETUNED_MODEL_PATH = 'mobilenetv2_model_finetuned.keras'

# Output files
INITIAL_PRED_CSV = 'mobpred.csv'
FINETUNED_PRED_CSV = 'mobfinepred.csv'

# --- 2. Prepare Label Encoder and Custom Objects ---
# Fit the LabelEncoder once on the training data to map indices to class names
le = LabelEncoder()
df_train = pd.read_csv(TRAIN_CSV_PATH)
le.fit(df_train['TARGET'])
print("✅ LabelEncoder is ready.")

# Define the custom metric object needed to load the models
custom_objects = {
    "f1_micro": metrics.F1Score(average='micro', name='f1_micro')
}

# --- 3. Get the list of test files ---
# Using sorted() ensures the order is consistent
test_image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(os.path.join(TEST_IMAGE_DIR, f))])
print(f"Found {len(test_image_files)} images in the test directory.")


# --- 4. Prediction Function ---
# A helper function to avoid repeating code
def generate_predictions(model, files, output_csv_path):
    print(f"\n--- Generating predictions for {output_csv_path} ---")
    image_ids = []
    predictions = []
    
    for image_file in tqdm(files, desc=f"Predicting for {os.path.basename(model.name)}"):
        img_path = os.path.join(TEST_IMAGE_DIR, image_file)
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr_img_preprocessed = preprocess_input(np.array(img))
        img_batch = np.expand_dims(arr_img_preprocessed, axis=0)
        
        pred_probs = model.predict(img_batch, verbose=0)
        predicted_index = np.argmax(pred_probs[0])
        predicted_label = le.inverse_transform([predicted_index])[0]
        
        image_ids.append(image_file)
        predictions.append(predicted_label)
        
    submission_df = pd.DataFrame({'ID': image_ids, 'TARGET': predictions})
    submission_df.to_csv(output_csv_path, index=False)
    print(f"✅ Submission file created successfully at: {output_csv_path}")

# --- 5. Run Predictions for Both Models ---

# Initial Model
try:
    initial_model = models.load_model(INITIAL_MODEL_PATH, custom_objects=custom_objects)
    generate_predictions(initial_model, test_image_files, INITIAL_PRED_CSV)
except Exception as e:
    print(f"❌ Error loading or predicting with initial model: {e}")

# Fine-Tuned Model
try:
    finetuned_model = models.load_model(FINETUNED_MODEL_PATH, custom_objects=custom_objects)
    generate_predictions(finetuned_model, test_image_files, FINETUNED_PRED_CSV)
except Exception as e:
    print(f"❌ Error loading or predicting with fine-tuned model: {e}")

print("\n--- All predictions complete! ---")