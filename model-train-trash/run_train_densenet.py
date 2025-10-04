import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- Changed for DenseNet121 ---
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
# ------------------------------

from tensorflow.keras import layers, models, callbacks, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
df = pd.read_csv("../dataset/train.csv")
print("train.csv loaded. Shape:", df.shape)

data = []
labels = []
IMG_SIZE = 224

for _, row in df.iterrows():
    img_path = os.path.join("../dataset/train", row['ID'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr_img = preprocess_input(np.array(img))
        data.append(arr_img)
        labels.append(row['TARGET'])

data = np.array(data)
labels = np.array(labels)

print("\nData processed. Image array shape:", data.shape)

# Encode labels as integers (no one-hot)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)
print(f"Labels encoded successfully. Number of classes: {NUM_CLASSES}")

# --- 2. Split Data ---
print("\n--- Step 2: Splitting Data ---")
X_train, X_val, y_train, y_val = train_test_split(
    data, labels_encoded,
    test_size=0.2, random_state=42,
    stratify=labels_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

# --- 3. Setup Data Augmentation ---
print("\n--- Step 3: Setting up Data Augmentation ---")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_gen = datagen.flow(X_train, y_train, batch_size=32)
print("Data Augmentation generator created.")

class_weights_array = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class Weights calculated:", class_weights)

# --- 4. Build the Model ---
print("\n--- Step 4: Building the DenseNet121 Model ---")
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Define metrics (F1 works with integer labels now ✅)
f1_micro = metrics.F1Score(average='micro', name="f1_micro")

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy', f1_micro]
)
print("Model compiled successfully.")
model.summary()

# --- 5. Setup Callbacks ---
print("\n--- Step 5: Setting up Callbacks ---")
checkpoint = callbacks.ModelCheckpoint(
    'densenet_model.keras', monitor='val_f1_micro',
    mode='max', save_best_only=True, verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor='val_f1_micro', mode='max',
    patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=3, min_lr=1e-6, verbose=1
)
print("Callbacks defined.")

# --- 6. Train the Model ---
print("\n--- Step 6: Starting Model Training ---")
history = model.fit(
    train_gen,
    validation_data=(X_val, y_val),
    epochs=50,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("✅ Training complete!")
