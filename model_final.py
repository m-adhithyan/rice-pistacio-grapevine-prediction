# ===================================================================
#
#       COMPLETE IMAGE CLASSIFICATION PIPELINE
#
# This script combines all steps:
# 1. Initial training of ResNet50
# 2. Fine-tuning of ResNet50
# 3. Initial training of EfficientNetB2
# 4. Fine-tuning of EfficientNetB2
# 5. Ensemble prediction using the fine-tuned models
# 
# ===================================================================

#Note:The dataset directory structure is assumed to be:
# dataset/
# ├── train/
# ├── test/
# ├── train.csv
# └── sample_submission.csv

import pandas as pd
import numpy as np
import os
from PIL import Image
import math
import tensorflow as tf
import tensorflow_addons as tfa
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models, callbacks, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Import model-specific components
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess_input, EfficientNetB2

# ===================================================================
# STAGE 1: ResNet50 - Initial Training (from resnet.py)
# ===================================================================
print("\n" + "="*50)
print(" STAGE 1: INITIAL TRAINING OF RESNET50")
print("="*50 + "\n")

# --- 1.1 Load and Preprocess Data for ResNet50 ---
print("--- Step 1.1: Loading and Preprocessing Data for ResNet50 ---")
df_r = pd.read_csv("../dataset/train.csv")
data_r = []
labels_r_list = []
IMG_SIZE_R = 224

for _, row in df_r.iterrows():
    img_path = os.path.join("../dataset/train", row['ID'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE_R, IMG_SIZE_R))
        arr_img = resnet_preprocess_input(np.array(img))
        data_r.append(arr_img)
        labels_r_list.append(row['TARGET'])

data_r = np.array(data_r)
labels_r = np.array(labels_r_list)
le_r = LabelEncoder()
labels_encoded_r = le_r.fit_transform(labels_r)
labels_categorical_r = to_categorical(labels_encoded_r, num_classes=len(le_r.classes_))
NUM_CLASSES_R = len(le_r.classes_)

# --- 1.2 Split Data ---
print("\n--- Step 1.2: Splitting Data ---")
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    data_r, labels_categorical_r, test_size=0.2, random_state=42, stratify=labels_categorical_r
)

# --- 1.3 Setup Data Augmentation ---
print("\n--- Step 1.3: Setting up Data Augmentation ---")
datagen_r = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', brightness_range=[0.8, 1.2]
)
train_gen_r = datagen_r.flow(X_train_r, y_train_r, batch_size=32)

def train_generator_wrapper_r():
    for x, y in train_gen_r: yield x, y

train_ds_r = tf.data.Dataset.from_generator(
    train_generator_wrapper_r,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE_R, IMG_SIZE_R, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_CLASSES_R), dtype=tf.float32)
    )
)
class_weights_r = {i: w for i, w in enumerate(compute_class_weight('balanced', classes=np.unique(labels_r), y=labels_r))}

# --- 1.4 Build the Model ---
print("\n--- Step 1.4: Building the ResNet50 Model ---")
base_model_r = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE_R, IMG_SIZE_R, 3))
base_model_r.trainable = False
model_r = models.Sequential([
    base_model_r,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES_R, activation='softmax')
])
f1_micro_r = metrics.F1Score(average='micro', name="f1_micro")
model_r.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_micro_r])

# --- 1.5 Train the Model ---
print("\n--- Step 1.5: Starting ResNet50 Initial Training ---")
checkpoint_r = callbacks.ModelCheckpoint('resnet_model_robust.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1)
early_stop_r = callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1)
reduce_lr_r = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
history_r = model_r.fit(
    train_ds_r, validation_data=(X_val_r, y_val_r), epochs=50,
    steps_per_epoch=len(X_train_r) // 32, validation_steps=len(X_val_r) // 32,
    class_weight=class_weights_r, callbacks=[checkpoint_r, early_stop_r, reduce_lr_r]
)
print("✅ ResNet50 Initial Training complete!")


# ===================================================================
# STAGE 2: ResNet50 - Fine-Tuning (from resnetfinetune.py)
# ===================================================================
print("\n" + "="*50)
print(" STAGE 2: FINE-TUNING RESNET50")
print("="*50 + "\n")

# --- 2.1 Load the Pre-Trained Model ---
print("\n--- Step 2.1: Loading the Best Model from Initial Training ---")
f1_micro_ft_r = metrics.F1Score(average='micro', name="f1_micro")
model_ft_r = models.load_model('resnet_model_robust.keras', custom_objects={'F1Score': f1_micro_ft_r})

# --- 2.2 Unfreeze Layers for Fine-Tuning ---
print("\n--- Step 2.2: Unfreezing top layers of the base model ---")
base_model_ft_r = model_ft_r.layers[0]
base_model_ft_r.trainable = True
fine_tune_at_r = 143
for layer in base_model_ft_r.layers[:fine_tune_at_r]:
    layer.trainable = False

# --- 2.3 Re-compile Model for Fine-Tuning ---
print("\n--- Step 2.3: Re-compiling for Fine-Tuning ---")
optimizer_ft_r = tf.keras.optimizers.Adam(learning_rate=1e-5)
model_ft_r.compile(optimizer=optimizer_ft_r,
                 loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                 metrics=['accuracy', f1_micro_ft_r])

# --- 2.4 Start Fine-Tuning ---
print("\n--- Step 2.4: Starting ResNet50 Fine-Tuning ---")
checkpoint_ft_r = callbacks.ModelCheckpoint('resnet_model_finetuned.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1)
early_stop_ft_r = callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1)
reduce_lr_ft_r = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
history_ft_r = model_ft_r.fit(
    train_ds_r, validation_data=(X_val_r, y_val_r), epochs=20,
    steps_per_epoch=len(X_train_r) // 32, validation_steps=len(X_val_r) // 32,
    class_weight=class_weights_r, callbacks=[checkpoint_ft_r, early_stop_ft_r, reduce_lr_ft_r]
)
print("✅ ResNet50 Fine-Tuning complete!")


# ===================================================================
# STAGE 3: EfficientNetB2 - Initial Training (from train_model_b2.py)
# ===================================================================
print("\n" + "="*50)
print(" STAGE 3: INITIAL TRAINING OF EFFICIENTNETB2")
print("="*50 + "\n")

# --- 3.1 Load and Preprocess Data for EfficientNetB2 ---
print("--- Step 3.1: Loading and Preprocessing Data for EfficientNetB2 ---")
df_e = pd.read_csv("../dataset/train.csv")
data_e = []
labels_e_list = []
IMG_SIZE_E = 260

for _, row in df_e.iterrows():
    img_path = os.path.join("../dataset/train", row['ID'])
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE_E, IMG_SIZE_E))
        arr_img = effnet_preprocess_input(np.array(img))
        data_e.append(arr_img)
        labels_e_list.append(row['TARGET'])

data_e = np.array(data_e)
labels_e = np.array(labels_e_list)
le_e = LabelEncoder()
labels_encoded_e = le_e.fit_transform(labels_e)
labels_categorical_e = to_categorical(labels_encoded_e, num_classes=len(le_e.classes_))
NUM_CLASSES_E = len(le_e.classes_)

# --- 3.2 Split Data ---
print("\n--- Step 3.2: Splitting Data ---")
X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(
    data_e, labels_categorical_e, test_size=0.2, random_state=42, stratify=labels_categorical_e
)

# --- 3.3 Setup Data Augmentation ---
print("\n--- Step 3.3: Setting up Data Augmentation ---")
datagen_e = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', brightness_range=[0.8, 1.2]
)
train_gen_e = datagen_e.flow(X_train_e, y_train_e, batch_size=32)

def train_generator_wrapper_e():
    for x, y in train_gen_e: yield x, y

train_ds_e = tf.data.Dataset.from_generator(
    train_generator_wrapper_e,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE_E, IMG_SIZE_E, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_CLASSES_E), dtype=tf.float32)
    )
)
class_weights_e = {i: w for i, w in enumerate(compute_class_weight('balanced', classes=np.unique(labels_e), y=labels_e))}

# --- 3.4 Build the Model ---
print("\n--- Step 3.4: Building the EfficientNetB2 Model ---")
base_model_e = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE_E, IMG_SIZE_E, 3))
base_model_e.trainable = False
model_e = models.Sequential([
    base_model_e,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES_E, activation='softmax')
])
f1_micro_e = metrics.F1Score(average='micro', name="f1_micro")
model_e.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_micro_e])

# --- 3.5 Train the Model ---
print("\n--- Step 3.5: Starting EfficientNetB2 Initial Training ---")
checkpoint_e = callbacks.ModelCheckpoint('b2_model_robust.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1)
early_stop_e = callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1)
reduce_lr_e = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
history_e = model_e.fit(
    train_ds_e, validation_data=(X_val_e, y_val_e), epochs=50,
    steps_per_epoch=len(X_train_e) // 32, validation_steps=len(X_val_e) // 32,
    class_weight=class_weights_e, callbacks=[checkpoint_e, early_stop_e, reduce_lr_e]
)
print("✅ EfficientNetB2 Initial Training complete!")


# ===================================================================
# STAGE 4: EfficientNetB2 - Fine-Tuning (from b2_finetune.py)
# ===================================================================
print("\n" + "="*50)
print(" STAGE 4: FINE-TUNING EFFICIENTNETB2")
print("="*50 + "\n")

# --- 4.1 Load the Pre-Trained Model ---
print("\n--- Step 4.1: Loading the Best Model from Initial Training ---")
f1_micro_ft_e = metrics.F1Score(average='micro', name="f1_micro")
model_ft_e = models.load_model('b2_model_robust.keras', custom_objects={'f1_micro': f1_micro_ft_e})

# --- 4.2 Prepare Model for Fine-Tuning ---
print("\n--- Step 4.2: Preparing Model for Fine-Tuning ---")
base_model_ft_e = model_ft_e.layers[0]
base_model_ft_e.trainable = True
for layer in base_model_ft_e.layers[:-20]:
    layer.trainable = False
optimizer_ft_e = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
model_ft_e.compile(optimizer=optimizer_ft_e,
                 loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                 metrics=['accuracy', f1_micro_ft_e])

# --- 4.3 Define Callbacks for Fine-Tuning ---
print("\n--- Step 4.3: Setting up Callbacks for Fine-Tuning ---")
steps_per_epoch_e = len(X_train_e) // 32
EPOCHS_FT_E = 20
WARMUP_EPOCHS_E = 3
lr_schedule_e = tf.keras.optimizers.schedules.CosineDecay(
    1e-5, decay_steps=steps_per_epoch_e * (EPOCHS_FT_E - WARMUP_EPOCHS_E),
    alpha=0.0, warmup_target=1e-5, warmup_steps=steps_per_epoch_e * WARMUP_EPOCHS_E
)
lr_scheduler_callback_e = tf.keras.callbacks.LearningRateScheduler(lr_schedule_e)
checkpoint_ft_e = callbacks.ModelCheckpoint('b2_model_finetuned.keras', monitor='val_f1_micro', mode='max', save_best_only=True, verbose=1)
early_stop_ft_e = callbacks.EarlyStopping(monitor='val_f1_micro', mode='max', patience=5, restore_best_weights=True, verbose=1)

# --- 4.4 Run the Fine-Tuning Process ---
print("\n--- Step 4.4: Starting Model Fine-Tuning ---")
history_ft_e = model_ft_e.fit(
    train_ds_e, validation_data=(X_val_e, y_val_e), epochs=EPOCHS_FT_E,
    steps_per_epoch=steps_per_epoch_e, validation_steps=len(X_val_e) // 32,
    class_weight=class_weights_e, callbacks=[checkpoint_ft_e, early_stop_ft_e, lr_scheduler_callback_e]
)
print("✅ EfficientNetB2 Fine-Tuning complete!")


# ===================================================================
# STAGE 5: Ensemble Prediction (from predict_ensemble.py)
# ===================================================================
print("\n" + "="*50)
print(" STAGE 5: ENSEMBLE PREDICTION")
print("="*50 + "\n")

# --- 5.1 Configuration ---
MODELS_TO_ENSEMBLE = {
    'resnet_finetuned': {
        'path': 'resnet_model_finetuned.keras',
        'img_size': 224,
        'preprocess_func': resnet_preprocess_input,
        'custom_objects': {'f1_micro': metrics.F1Score(average='micro', name="f1_micro")}
    },
    'efficientnet_b2_finetuned': {
        'path': 'b2_model_finetuned.keras',
        'img_size': 260,
        'preprocess_func': effnet_preprocess_input,
        'custom_objects': {
            'f1_micro': metrics.F1Score(average='micro', name="f1_micro"),
            'AdamW': tfa.optimizers.AdamW
        }
    },
}
TTA_STEPS = 5
BATCH_SIZE = 16
TEST_IMAGE_DIR = '../dataset/test'
OUTPUT_FILENAME = 'submission_ensemble.csv'

# --- 5.2 Load Models ---
print("\n--- Step 5.2: Loading Ensemble Models ---")
loaded_models = {}
for name, model_info in MODELS_TO_ENSEMBLE.items():
    print(f"-> Loading '{name}' from {model_info['path']}...")
    loaded_models[name] = load_model(model_info['path'], custom_objects=model_info['custom_objects'])
print("✅ All models loaded successfully.")

# --- 5.3 Setup Data and Label Encoder ---
df_test = pd.read_csv('../dataset/sample_submission.csv')
df_train_for_le = pd.read_csv('../dataset/train.csv')
le_final = LabelEncoder()
le_final.fit(df_train_for_le['TARGET'])
print("\n✅ Label Encoder setup complete.")

# --- 5.4 TTA and Ensemble Prediction ---
tta_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, zoom_range=0.15)
all_final_predictions = []
num_batches = math.ceil(len(df_test) / BATCH_SIZE)
print(f"\n🚀 Starting Ensemble TTA prediction on {len(df_test)} images...")

for i in range(num_batches):
    start_index = i * BATCH_SIZE
    end_index = min((i + 1) * BATCH_SIZE, len(df_test))
    batch_df = df_test.iloc[start_index:end_index]
    model_predictions_for_batch = []
    print(f"   -> Processing Batch {i+1}/{num_batches}...")

    for name, model in loaded_models.items():
        model_info = MODELS_TO_ENSEMBLE[name]
        IMG_SIZE = model_info['img_size']
        preprocess_func = model_info['preprocess_func']
        
        batch_data = np.array([
            preprocess_func(np.array(Image.open(os.path.join(TEST_IMAGE_DIR, row['ID'])).convert('RGB').resize((IMG_SIZE, IMG_SIZE))))
            for _, row in batch_df.iterrows()
        ])
        
        tta_preds = [model.predict(batch_data, verbose=0)]
        for _ in range(TTA_STEPS - 1):
            augmented_batch = next(tta_datagen.flow(batch_data, batch_size=len(batch_data), shuffle=False))
            tta_preds.append(model.predict(augmented_batch, verbose=0))
        model_predictions_for_batch.append(np.mean(tta_preds, axis=0))

    final_ensembled_preds = np.mean(model_predictions_for_batch, axis=0)
    all_final_predictions.append(final_ensembled_preds)

print("\n✅ Ensemble TTA Predictions for all batches generated.")

# --- 5.5 Create Submission File ---
print("\n--- Step 5.5: Creating Submission File ---")
final_predictions = np.vstack(all_final_predictions)
predicted_labels_encoded = np.argmax(final_predictions, axis=1)
predicted_labels = le_final.inverse_transform(predicted_labels_encoded)
submission_df = pd.DataFrame({'ID': df_test['ID'], 'TARGET': predicted_labels})
submission_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Final submission file '{OUTPUT_FILENAME}' created successfully!")
print("\nPreview of submission file:")
print(submission_df.head())
