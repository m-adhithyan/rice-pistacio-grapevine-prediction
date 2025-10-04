import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB1
from tensorflow.keras import layers, models, callbacks, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
df = pd.read_csv("../dataset/train.csv")
print("train.csv loaded. Shape:", df.shape)

data = []
labels = []
IMG_SIZE = 240 # EfficientNetB1 uses 240x240

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

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, num_classes=len(le.classes_))
NUM_CLASSES = len(le.classes_)
print("Labels encoded successfully.")

# --- 2. Split Data ---
print("\n--- Step 2: Splitting Data ---")
X_train, X_val, y_train, y_val = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# --- 3. Setup Data Augmentation and tf.data.Dataset ---
print("\n--- Step 3: Setting up Data Augmentation and Dataset ---")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)
train_gen = datagen.flow(X_train, y_train, batch_size=32)
print("Data Augmentation generator created.")

# Wrap the generator in a tf.data.Dataset to define the output signature
def train_generator_wrapper():
    for x, y in train_gen:
        yield x, y

train_ds = tf.data.Dataset.from_generator(
    train_generator_wrapper,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
    )
)
print("tf.data.Dataset created.")


class_weights_array = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class Weights calculated:", class_weights)

# --- 4. Build the Model ---
print("\n--- Step 4: Building the Model ---")
base_model = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

f1_micro = metrics.F1Score(average='micro', name="f1_micro")
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_micro])
print("Model compiled successfully.")

# --- 5. Setup Callbacks ---
print("\n--- Step 5: Setting up Callbacks ---")
checkpoint = callbacks.ModelCheckpoint('best_model_robust.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
print("Callbacks defined.")

# --- 6. Train the Model ---
print("\n--- Step 6: Starting Model Training ---")
steps_per_epoch = len(X_train) // 32
validation_steps = len(X_val) // 32

history = model.fit(
    train_ds,
    validation_data=(X_val, y_val),
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)
print("✅ Training complete!")

# --- 7. Evaluate the Final Model ---
print("\n--- Step 7: Evaluating Final Model on Validation Set ---")
val_loss, val_acc, val_f1 = model.evaluate(X_val, y_val)
print(f"✅ Final Validation Loss: {val_loss:.4f}")
print(f"✅ Final Validation Accuracy: {val_acc:.4f}")
print(f"✅ Final Validation F1 Micro: {val_f1:.4f}")