import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2S
from tensorflow.keras import layers, models, callbacks, metrics, optimizers

# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
df = pd.read_csv("../dataset/train.csv")
data = []
labels = []
IMG_SIZE = 384
BATCH_SIZE = 16 # Use a smaller batch size for large images

for _, row in df.iterrows():
    img_path = os.path.join("../dataset/train", row['ID'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr_img = preprocess_input(np.array(img))
        data.append(arr_img)
        labels.append(row['TARGET'])

data = np.array(data)
labels = np.array(labels)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)
NUM_CLASSES = len(le.classes_)

# --- 2. Split Data ---
X_train, X_val, y_train, y_val = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
)

# --- 3. Setup Data Augmentation and tf.data.Dataset ---
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', brightness_range=[0.8, 1.2]
)
train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
def train_generator_wrapper():
    for x, y in train_gen: yield x, y
train_ds = tf.data.Dataset.from_generator(
    train_generator_wrapper,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
    )
)

class_weights = {i: w for i, w in enumerate(compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded))}

# --- 4. Build the Model ---
base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=optimizers.AdamW(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', metrics.F1Score(average='micro', name="f1_micro")])

# --- 5. Train the Model ---
callbacks_list = [
    callbacks.ModelCheckpoint('new_efficientnetv2s_model.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

model.fit(
    train_ds,
    validation_data=(X_val, y_val),
    epochs=50,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks_list
)