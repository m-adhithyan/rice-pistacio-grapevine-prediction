import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, metrics

# --- 1. CONFIGURE FOR EACH MODEL ---
# You will change this section for each model before running the script.
# Example for MobileNetV2 is shown below.

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
MODEL_PATH = 'mobilenetv2_model_finetuned.keras'
OUTPUT_NPY_PATH = 'mobilenet_finetuned_probabilities.npy'
IMG_SIZE = 224

# --- 2. SETUP (No changes needed here) ---
TEST_IMAGE_DIR = "../dataset/test"
TRAIN_CSV_PATH = "../dataset/train.csv"

le = LabelEncoder()
df_train = pd.read_csv(TRAIN_CSV_PATH)
le.fit(df_train['TARGET'])

custom_objects = { "f1_micro": metrics.F1Score(average='micro', name='f1_micro') }
test_image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(os.path.join(TEST_IMAGE_DIR, f))])

# --- 3. LOAD MODEL AND PREDICT ---
print(f"Loading model: {MODEL_PATH}")
model = models.load_model(MODEL_PATH, custom_objects=custom_objects)

all_probabilities = []
for image_file in tqdm(test_image_files, desc=f"Predicting with {os.path.basename(MODEL_PATH)}"):
    img_path = os.path.join(TEST_IMAGE_DIR, image_file)
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr_img_preprocessed = preprocess_input(np.array(img))
    img_batch = np.expand_dims(arr_img_preprocessed, axis=0)
    
    pred_probs = model.predict(img_batch, verbose=0)
    all_probabilities.append(pred_probs[0])

# --- 4. SAVE PROBABILITIES ---
np.save(OUTPUT_NPY_PATH, np.array(all_probabilities))
print(f"✅ Probabilities saved successfully to: {OUTPUT_NPY_PATH}")