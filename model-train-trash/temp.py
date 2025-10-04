import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models

import tensorflow_addons as tfa # NEW IMPORT

# --- 1. Configuration for your EfficientNetV2S model ---
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

MODEL_PATH = 'efficientnetv2s_model.keras'
OUTPUT_NPY_PATH = 'efficientnetv2s_probabilities.npy'
IMG_SIZE = 384

# --- 2. SETUP ---
TEST_IMAGE_DIR = "../dataset/test"
TRAIN_CSV_PATH = "../dataset/train.csv"

le = LabelEncoder()
df_train = pd.read_csv(TRAIN_CSV_PATH)
le.fit(df_train['TARGET'])

# CHANGED: Use tfa.metrics.F1Score
# The F1Score metric is needed to load the model
custom_objects = { "f1_micro": tfa.metrics.F1Score(average='micro', name='f1_micro') }

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
print(f"✅ Probabilities for EfficientNetV2S saved successfully to: {OUTPUT_NPY_PATH}")