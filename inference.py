import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import polars as pl
from scipy.optimize import linear_sum_assignment

from model import TwoBranchModel
from helpers import remove_gravity_from_acc, calculate_angular_velocity_from_quat, calculate_angular_distance, pad_sequences_torch


# %%
# Configuration
PRETRAINED_DIR = Path("/kaggle/input/cmi-dbsd-models")
EXPORT_DIR = Path("./")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]

# Load model
MODELS = [f'gesture_two_branch_fold{i}.pth' for i in range(5)]

models = []
for path in MODELS:
    checkpoint = torch.load(PRETRAINED_DIR / path, map_location=device)
    
    # Instantiate model using structural params from checkpoint
    model = TwoBranchModel(
        checkpoint['pad_len'], 
        checkpoint['imu_dim'], 
        checkpoint['tof_dim'], 
        checkpoint['n_classes']
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)


# %%
SUBJECT_HISTORY = {}

FINAL_PREDICTIONS = {}
gesture_classes = None
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    A stateful prediction function with a global optimal assignment post-processing.
    Notes:
    - The Kaggle evaluation API calls this per sequence (multiple sequences per subject).
    - We maintain log-probability vectors for all processed sequences of each subject, and
      use the Hungarian algorithm to perform a global optimal matching on the (sequences × classes) matrix,
      assigning each visible sequence to a non-conflicting class (maximizing total log probability).
    - This suppresses the tendency to repeatedly predict the same class for similar sequences of a subject,
      improving fine-grained class stability (especially when each subject spans multiple actions).
    """
    global gesture_classes, SUBJECT_HISTORY, FINAL_PREDICTIONS  # Use module-level globals (persist across calls)

    # --- Initialization / reset logic ---
    if gesture_classes is None:
        print("First call of this submission run. Initializing...")
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
        SUBJECT_HISTORY = {}
        FINAL_PREDICTIONS = {}

    # --- 1. Feature engineering ---
    df_seq = sequence.to_pandas()
    subject_id = df_seq['subject'].iloc[0]
    sequence_id = df_seq['sequence_id'].iloc[0]
    
    df_demo = demographics.to_pandas()
    df_seq = df_seq.merge(df_demo[['subject', 'handedness']], on='subject', how='left')
    
    handedness = df_seq['handedness'].iloc[0]
    if handedness == 0:
        # Swap ToF/Thermopile sensor 3 and 5
        cols_3 = [c for c in df_seq.columns if any(p in c for p in ['tof_3', 'thm_3'])]
        cols_5 = [c for c in df_seq.columns if any(p in c for p in ['tof_5', 'thm_5'])]
        
        cols_3.sort()
        cols_5.sort()
        
        if len(cols_3) == len(cols_5):
            temp_cols_3_data = df_seq[cols_3].copy()
            df_seq[cols_3] = df_seq[cols_5]
            df_seq[cols_5] = temp_cols_3_data
        
        negate_cols = ['acc_x', 'rot_y', 'rot_z']
        df_seq[negate_cols] *= -1
    
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'], df_seq['linear_acc_y'], df_seq['linear_acc_z'] = linear_accel[:, 0], linear_accel[:, 1], linear_accel[:, 2]
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'], df_seq['angular_vel_y'], df_seq['angular_vel_z'] = angular_vel[:, 0], angular_vel[:, 1], angular_vel[:, 2]
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)

    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = df_seq[pixel_cols].replace(-1, np.nan)
        df_seq[f'tof_{i}_mean'], df_seq[f'tof_{i}_std'], df_seq[f'tof_{i}_min'], df_seq[f'tof_{i}_max'] = tof_data.mean(axis=1), tof_data.std(axis=1), tof_data.min(axis=1), tof_data.max(axis=1)

    mat_unscaled = df_seq[feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    mat = scaler.transform(mat_unscaled)
    pad = pad_sequences_torch([mat], maxlen=pad_len, padding='pre', truncating='pre')
    

    # --- 2. Model prediction: average logits across folds -> compute log-softmax ---
    with torch.no_grad():
        x = torch.FloatTensor(pad).to(device)
        all_logits = []
        for model in models:
            model.eval()
            logits = model(x)[0]
            all_logits.append(logits)
        
        avg_logits = torch.stack(all_logits).mean(dim=0)
        log_probs = F.log_softmax(avg_logits, dim=1).cpu().numpy().flatten()
    

    # --- 3. Accumulate this sequence's prediction history (grouped by subject) ---
    if subject_id not in SUBJECT_HISTORY:
        SUBJECT_HISTORY[subject_id] = []
        
    if not any(d['seq_id'] == sequence_id for d in SUBJECT_HISTORY[subject_id]):
        SUBJECT_HISTORY[subject_id].append({'seq_id': sequence_id, 'log_probs': log_probs})


    # --- 4. Run a global optimal assignment over all sequences seen for this subject ---
    subject_history = SUBJECT_HISTORY[subject_id]
    num_sequences_so_far = len(subject_history)
    num_labels = len(log_probs)

    # a) Build cost matrix (N × C): use -log_prob as cost (minimize cost == maximize total log prob)
    cost_matrix = np.zeros((num_sequences_so_far, num_labels))
    for i in range(num_sequences_so_far):
        cost_matrix[i, :] = -subject_history[i]['log_probs']
    
    # b) Hungarian algorithm solves the linear assignment: assign a unique class to each sequence minimizing total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # c) Update final prediction for all known sequences using the assignment
    for i in range(num_sequences_so_far):
        seq_info = subject_history[i]
        assigned_label_index = col_ind[i]
        final_gesture_name = gesture_classes[assigned_label_index]
        
        final_gesture_name = final_gesture_name.split('_')[1]
        FINAL_PREDICTIONS[seq_info['seq_id']] = final_gesture_name
        
        
    # --- 5. Return the final prediction for the current sequence (from the global cache) ---
    return FINAL_PREDICTIONS[sequence_id]


# %%
import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )
