import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- We need the specific preprocess_input for the model we are fine-tuning ---
from tensorflow.keras.applications.resnet import preprocess_input
# -----------------------------------------------------------------------------

from tensorflow.keras import layers, models, callbacks, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This script assumes 'resnet_model_robust.keras' exists from a previous training run.

# ===================================================================
#
#       STANDALONE SCRIPT FOR FINE-TUNING A TRAINED MODEL
#
# ===================================================================


# --- Step 1: Load and Preprocess Data ---
# We need to reload the data to create the data generators and have the validation set for evaluation.
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

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, num_classes=len(le.classes_))
NUM_CLASSES = len(le.classes_)
print("Labels encoded successfully.")

# --- 2. Split Data ---
# It's CRITICAL to use the same random_state to ensure the validation set is identical.
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


# --- Step 4: Load the Pre-Trained Model ---
print("\n--- Step 4: Loading the Best Model from Initial Training ---")
# Define the custom metric function needed for loading the model
f1_micro = metrics.F1Score(average='micro', name="f1_micro")
model = models.load_model(
    'resnet_model_robust.keras',
    custom_objects={'F1Score': f1_micro}
)
print("Model loaded successfully.")


# --- Step 5: Unfreeze Layers for Fine-Tuning ---
print("\n--- Step 5: Unfreezing top layers of the base model ---")
base_model = model.layers[0]
base_model.trainable = True

print(f"Number of layers in the base model: {len(base_model.layers)}")
# For ResNet50, a good starting point is to unfreeze from block 5 onwards (layer 143).
fine_tune_at = 143
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
print(f"Base model unfrozen. First {fine_tune_at} layers are frozen.")


# --- Step 6: Re-compile Model for Fine-Tuning ---
print("\n--- Step 6: Re-compiling for Fine-Tuning ---")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) # Very low learning rate
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_micro])
print("Model re-compiled.")
model.summary()


# --- Step 7: Define Callbacks and Start Fine-Tuning ---
print("\n--- Step 7: Starting Fine-Tuning ---")
checkpoint_finetune = callbacks.ModelCheckpoint(
    'resnet_model_finetuned.keras',
    monitor='val_f1_micro',
    mode='max',
    save_best_only=True,
    verbose=1
)
early_stop_finetune = callbacks.EarlyStopping(
    monitor='val_f1_micro',
    mode='max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_finetune = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

steps_per_epoch = len(X_train) // 32
validation_steps = len(X_val) // 32

history_fine_tune = model.fit(
    train_ds,
    validation_data=(X_val, y_val),
    epochs=20, # Train for 20 epochs
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[checkpoint_finetune, early_stop_finetune, reduce_lr_finetune]
)
print("✅ Fine-tuning complete!")


# --- Step 8: Evaluate the Final Fine-Tuned Model ---
print("\n--- Step 8: Evaluating Final Fine-Tuned Model on Validation Set ---")
val_loss, val_acc, val_f1 = model.evaluate(X_val, y_val)
print(f"✅ Final Fine-Tuned Validation Loss: {val_loss:.4f}")
print(f"✅ Final Fine-Tuned Validation Accuracy: {val_acc:.4f}")
print(f"✅ Final Fine-Tuned Validation F1 Micro: {val_f1:.4f}")