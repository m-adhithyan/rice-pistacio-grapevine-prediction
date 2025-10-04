import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata

# --- 1. SETUP FILE PATHS ---
PROB_FILES = [
    'resnet_probabilities.npy',
    'efficientnetv2s_probabilities.npy'
]
FINAL_SUBMISSION_FILE = 'final_ensemble_submission_v4_2.csv' # New name for this version

TRAIN_CSV_PATH = "../dataset/train.csv"
TEST_IMAGE_DIR = "../dataset/test"

# --- 2. ENSEMBLE PREDICTIONS ---
print("Loading model prediction probabilities...")
all_probs = np.stack([np.load(f) for f in PROB_FILES])

print("Performing rank averaging...")
# This converts each model's probability output to ranks
all_ranks = np.array([rankdata(-p, method='ordinal', axis=1) for p in all_probs])

# Average the ranks across the models
averaged_ranks = np.mean(all_ranks, axis=0)

# Get the index of the class with the LOWEST average rank (rank 1 is the best)
final_predictions_indices = np.argmin(averaged_ranks, axis=1)

# --- 3. CREATE FINAL SUBMISSION FILE ---
print(f"Creating final submission file at: {FINAL_SUBMISSION_FILE}")
df_train = pd.read_csv(TRAIN_CSV_PATH)
le = LabelEncoder()
le.fit(df_train['TARGET'])

final_labels = le.inverse_transform(final_predictions_indices)
test_image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(os.path.join(TEST_IMAGE_DIR, f))])

submission_df = pd.DataFrame({
    'ID': test_image_files,
    'TARGET': final_labels
})

submission_df.to_csv(FINAL_SUBMISSION_FILE, index=False)
print("✅ Final rank-ensembled submission file created!")