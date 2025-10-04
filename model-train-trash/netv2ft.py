import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# --- We need the specific preprocess_input for the model we are fine-tuning ---
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# -----------------------------------------------------------------------------

# ===================================================================
#
#       STANDALONE SCRIPT FOR FINE-TUNING EFFICIENTNETV2
#       (MEMORY EFFICIENT VERSION)
#
# This script assumes 'efficientnetv2s_model.keras' exists from a previous
# training run. It loads data from disk on-the-fly to save RAM.
#
# ===================================================================

# --- 1. SETUP & CONFIGURATION ---
IMAGE_DIR = '../dataset/train'
CSV_PATH = '../dataset/train.csv'
INITIAL_MODEL_PATH = 'efficientnetv2s_model.keras'
FINETUNED_MODEL_PATH = 'netv2s_finetuned.keras'

IMG_SIZE = 384
# --- MODIFICATION: Reduced batch size to lower memory usage ---
BATCH_SIZE = 8
# -------------------------------------------------------------
NUM_EPOCHS = 20
FINETUNE_LR = 1e-5

# --- 2. DATA PATH & LABEL PREPARATION (NO IMAGE LOADING YET) ---
print("--- Step 1: Preparing File Paths and Labels ---")
df = pd.read_csv(CSV_PATH)
print(f"{CSV_PATH} loaded. Shape: {df.shape}")

# Create full file paths and get labels
image_paths = [os.path.join(IMAGE_DIR, fname) for fname in df['ID']]
labels = df['TARGET'].values

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = tf.keras.utils.to_categorical(labels_encoded)
NUM_CLASSES = len(label_encoder.classes_)
print("Labels encoded successfully.")


# --- 3. SPLIT DATA (PATHS AND LABELS ONLY) ---
# We split the paths and labels, not the actual image data.
print("\n--- Step 2: Splitting Data Paths ---")
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    image_paths, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
)
print(f"Training set: {len(X_train_paths)} samples")
print(f"Validation set: {len(X_val_paths)} samples")


# --- 4. CREATE MEMORY-EFFICIENT TF.DATA.DATASET ---
print("\n--- Step 3: Setting up Memory-Efficient Dataset ---")

# This function will be mapped to our dataset to load images from disk on-the-fly.
def load_and_preprocess_from_path(path, label):
    # Read the file from disk
    img = tf.io.read_file(path)
    # Decode, resize, and preprocess the image
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(img)
    return img, label

# Create the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
train_dataset = train_dataset.map(load_and_preprocess_from_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val))
val_dataset = val_dataset.map(load_and_preprocess_from_path, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("tf.data.Dataset objects created to load images on-the-fly.")

# Calculate class weights
class_weights_array = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Class Weights calculated:", class_weights_dict)


# --- 5. LOAD THE PRE-TRAINED MODEL ---
print("\n--- Step 4: Loading the Best Model from Initial Training ---")
f1_metric = tf.keras.metrics.F1Score(average='micro', name="f1_micro")
model = tf.keras.models.load_model(
    INITIAL_MODEL_PATH,
    custom_objects={'F1Score': f1_metric}
)
print("Model loaded successfully.")


# --- 6. UNFREEZE LAYERS FOR FINE-TUNING ---
print("\n--- Step 5: Unfreezing top layers of the base model ---")
base_model = model.layers[0]
base_model.trainable = True

print(f"Number of layers in the base model: {len(base_model.layers)}")
fine_tune_at = -30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
print(f"Base model unfrozen. The last {-fine_tune_at} layers are now trainable.")


# --- 7. RE-COMPILE MODEL FOR FINE-TUNING ---
print("\n--- Step 6: Re-compiling for Fine-Tuning ---")
optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', f1_metric])
print("Model re-compiled with a very low learning rate.")
model.summary()


# --- 8. DEFINE CALLBACKS AND START FINE-TUNING ---
print("\n--- Step 7: Starting Fine-Tuning ---")
checkpoint_finetune = tf.keras.callbacks.ModelCheckpoint(
    FINETUNED_MODEL_PATH,
    monitor='val_f1_micro',
    mode='max',
    save_best_only=True,
    verbose=1
)
early_stop_finetune = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_micro',
    mode='max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_finetune = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history_fine_tune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=NUM_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_finetune, early_stop_finetune, reduce_lr_finetune]
)
print("✅ Fine-tuning complete!")


# --- 9. EVALUATE THE FINAL FINE-TUNED MODEL ---
print("\n--- Step 8: Evaluating Final Fine-Tuned Model on Validation Set ---")
results = model.evaluate(val_dataset, verbose=0)
print(f"✅ Final Fine-Tuned Validation Loss: {results[0]:.4f}")
print(f"✅ Final Fine-Tuned Validation Accuracy: {results[1]:.4f}")
print(f"✅ Final Fine-Tuned Validation F1 Micro: {results[2]:.4f}")