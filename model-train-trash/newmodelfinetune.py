import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, callbacks, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa

# --- 1. Load and Prepare Data for Fine-Tuning ---
print("--- Step 1: Loading and Preparing Data ---")
df = pd.read_csv("../dataset/train.csv")

data = []
labels = []
IMG_SIZE = 240
BATCH_SIZE = 32

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
labels_categorical = to_categorical(labels_encoded, num_classes=len(le.classes_))
NUM_CLASSES = len(le.classes_)

X_train, X_val, y_train, y_val = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
)

datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', brightness_range=[0.8, 1.2]
)
train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

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

class_weights_array = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: w for i, w in enumerate(class_weights_array)}

# ⭐ FIX: Define steps_per_epoch here, before it's needed by the callbacks
steps_per_epoch = len(X_train) // BATCH_SIZE
validation_steps = len(X_val) // BATCH_SIZE

print("✅ Data loading and preparation complete.")

# --- 2. Load the Pre-trained Model ---
print("\n--- Step 2: Loading the Pre-Trained Model ---")
f1_micro = metrics.F1Score(average='micro', name="f1_micro")
try:
    model = models.load_model('best_model_robust.keras', custom_objects={'f1_micro': f1_micro})
    print("✅ Loaded 'best_model_robust.keras' for fine-tuning.")
except Exception as e:
    print(f"❌ Could not load 'best_model_robust.keras'. Make sure initial training was successful. Error: {e}")
    exit()

# --- 3. Prepare Model for Fine-Tuning ---
print("\n--- Step 3: Preparing Model for Fine-Tuning ---")
base_model = model.layers[0]
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

optimizer = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_micro])
print("✅ Model re-compiled for fine-tuning.")

# --- 4. Define Callbacks for Fine-Tuning ---
print("\n--- Step 4: Setting up Callbacks for Fine-Tuning ---")
initial_learning_rate = 1e-5
EPOCHS_FT = 20
WARMUP_EPOCHS = 3

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=steps_per_epoch * (EPOCHS_FT - WARMUP_EPOCHS),
    alpha=0.0,
    warmup_target=initial_learning_rate,
    warmup_steps=steps_per_epoch * WARMUP_EPOCHS
)
lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

checkpoint_ft = callbacks.ModelCheckpoint('best_model_finetuned.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1)
early_stop_ft = callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1)
print("✅ Callbacks defined.")

# --- 5. Run the Fine-Tuning Process ---
print("\n--- Step 5: Starting Model Fine-Tuning ---")
history_finetune = model.fit(
    train_ds,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FT,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[checkpoint_ft, early_stop_ft, lr_scheduler_callback]
)
print("✅ Fine-tuning complete!")

# --- 6. Evaluate the Fine-Tuned Model ---
print("\n--- Step 6: Evaluating Final Fine-Tuned Model on Validation Set ---")
val_loss, val_acc, val_f1 = model.evaluate(X_val, y_val)
print(f"✅ Final Fine-Tuned Validation Loss: {val_loss:.4f}")
print(f"✅ Final Fine-Tuned Validation Accuracy: {val_acc:.4f}")
print(f"✅ Final Fine-Tuned Validation F1 Micro: {val_f1:.4f}")