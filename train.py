"""
• Use the Helios wrist device's multimodal sensors (IMU, Thermopile, ToF) to distinguish:
    (1) BFRB-like vs non-target
    (2) Specific BFRB-like gesture classes
• Evaluation: custom online metric (a variant of macro F1), equal-weight average of:
    - Binary F1: whether the gesture is target (BFRB-like)
    - Multiclass macro F1: merge all non-target gestures into a single 'non_target' class
"""

import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from timm.scheduler import CosineLRScheduler
import sys

from tqdm import tqdm

from functools import partial
from pytorch_metric_learning import losses, miners

from metric import CompetitionMetric
from model import TwoBranchModel
from dataset import CMI3Dataset, Augment, augment_left_handed_sequence, mixup_collate_fn
from helpers import set_seed
from helpers import remove_gravity_from_acc, calculate_angular_velocity_from_quat, calculate_angular_distance, pad_sequences_torch


# ================================
# Global configuration
# ================================
RAW_DIR = Path("data")       # Path to raw CSVs (train/test and demographics)
EXPORT_DIR = Path("artifacts")  # Output artifacts path (scaler, class mapping, model weights, etc.)
BATCH_SIZE = 64                 # Batch size
PAD_PERCENTILE = 100            # Sequence padding length (fixed 100 here; percentile also possible)
maxlen = PAD_PERCENTILE
LR_INIT = 1e-3                  # Initial learning rate (for AdamW)
WD = 3e-3                       # Weight decay (L2 regularization)
MASKING_PROB = 0.25             # Probability to randomly mask ToF/THM in collate (train gate robustness)
PATIENCE = 30                   # Early stopping patience (epochs)
FOLDS = 5                       # Number of folds for K-fold CV

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Prefer GPU if available
print(f"▶ imports ready · pytorch {torch.__version__} · device: {device}")  # Print environment info


# ================================
# Training main pipeline
# ================================
set_seed(42)
print("▶ TRAIN MODE – loading dataset …")
# Read main training table + demographics (handedness used for mirroring augmentation)
df = pd.read_csv(RAW_DIR / "train.csv")  # Read training data
df_demo = (pd.read_csv(RAW_DIR / "train_demographics.csv"))[['subject', 'handedness']]  # Read 'subject' and 'handedness' from demographics
df = df.merge(df_demo, on='subject', how='left')  # Merge to attach handedness per sequence

# Filter anomalous subjects; per forum discussion, their devices may have been worn upside down
df = df[~df['subject'].isin(["SUBJ_045235", "SUBJ_019262"])].reset_index(drop=True) 

print("  Transform the left hand into right hand")  # Log mirroring step
# Mirror left-handed sequences (handedness==0) to 'right-hand' semantics for alignment
df = df.groupby('sequence_id', group_keys=False).apply(augment_left_handed_sequence)

# ========== Sequence-level label construction ==========
# Concatenate orientation + gesture + initial_behavior as a composite label for stratification/finer-grained learning
df_seq_id = pd.DataFrame()  # Container to store per-sequence metadata

for seq_id, seq in tqdm(df.groupby('sequence_id')):  # Iterate each sequence
    orientation = seq['orientation'].iloc[0]  # Posture (e.g., lying/sitting)
    gesture = seq['gesture'].iloc[0]  # Gesture name
    initial_behaviors = seq['behavior'].iloc[0]  # Initial phase (transition/action)
    subject = seq['subject'].iloc[0]  # Subject ID

    temp_df = pd.DataFrame({
        'seq_id': seq_id,
        'orientation': orientation,
        'gesture': gesture,
        'initial_behavior': initial_behaviors,
        'subject': subject
    }, index=[0])  # Summarize as a single row
    df_seq_id = pd.concat([df_seq_id, temp_df], ignore_index=True)  # Append to the collection
df_seq_id['label'] = df_seq_id['orientation'] + '_' + df_seq_id['gesture'] + '_' + df_seq_id['initial_behavior']  # Build composite label

# Count composite label types
print(f"Number of unique labels: {df_seq_id['label'].nunique()}")  # Print number of classes

# Encode composite labels and save class order
le = LabelEncoder()  # Instantiate encoder
label_array = le.fit_transform(df_seq_id['label'])  # Fit and transform composite labels
os.makedirs(EXPORT_DIR, exist_ok=True)
np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)  # Save class list (for inference mapping)


# ========== Feature columns ==========
meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter', 'handedness'}  # Metadata columns
feature_cols_meta = [c for c in df.columns if c not in meta_cols]  # Candidate feature columns (initial)

print("  Removing gravity and calculating linear acceleration features…")  # Announce gravity removal
# Per-sequence: remove gravity to obtain linear acceleration
linear_accel_list = []  # Store linear acceleration per sequence
for _, group in df.groupby('sequence_id'):  # Iterate sequences
    acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]  # Three-axis acceleration
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]  # Quaternions
    linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)  # Remove gravity
    linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))  # To DF with preserved index
df_linear_accel = pd.concat(linear_accel_list)  # Concatenate all sequences

df = pd.concat([df, df_linear_accel], axis=1)  # Append linear acceleration features back

print("  Calculating angular velocity from quaternion derivatives…")  # Announce angular velocity calculation
# Per-sequence: derive angular velocity from quaternions
angular_vel_list = []  # Store per-sequence angular velocity
for _, group in df.groupby('sequence_id'):
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]  # Quaternions
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)  # Compute angular velocity
    angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))  # Collect
df_angular_vel = pd.concat(angular_vel_list)  # Concatenate all sequences

df = pd.concat([df, df_angular_vel], axis=1)  # Merge back to main frame

print("  Calculating angular distance between successive quaternions…")  # Announce angular distance calculation
# Per-sequence: angular distance between adjacent poses
angular_distance_list = []  # Container
for _, group in df.groupby('sequence_id'):
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]  # Quaternions
    angular_dist_group = calculate_angular_distance(rot_data_group)  # Angular distance
    angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))  # Collect
df_angular_distance = pd.concat(angular_distance_list)  # Concatenate

df = pd.concat([df, df_angular_distance], axis=1)  # Merge back to main frame

# IMU raw + engineered features
imu_cols_base = [c for c in feature_cols_meta if not (c.startswith('thm_') or c.startswith('tof_'))]  # Filter non-ToF/THM columns (IMU base)
imu_engineered_features = [
    'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance',
]  # Engineered IMU-derived features
imu_cols = imu_cols_base + imu_engineered_features  # Combine into full IMU feature set

# ToF/THM: raw thermopiles + ToF aggregated stats (each ToF 8x8=64 pixels, columns tof_i_v0..v63)
tof_aggregated_cols_template = []  # Per-ToF aggregated stats column-name template
for i in range(1, 6):
    tof_aggregated_cols_template.extend([f'tof_{i}_{stat}' for stat in ['mean', 'std', 'min', 'max']])  # Generate tof_i_mean/std/min/max

thm_cols_original = [c for c in df.columns if c.startswith('thm_')]  # Original 5 thermopile columns
tof_cols = thm_cols_original + tof_aggregated_cols_template  # Full ToF/THM feature set
feature_cols = imu_cols + tof_cols  # Final input feature columns
print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")  # Print dimensions


# ========== Build sequence matrices ==========
seq_gp = df.groupby('sequence_id')  # Group by sequence
all_steps_for_scaler_list = []  # Aggregate all steps to fit scaler (concatenate time steps)
X_list, y_list, id_list, hand_list, lens = [], [], [], [],[]  # Store features/labels/IDs/handedness/lengths
for idx, (seq_id, seq) in tqdm(enumerate(seq_gp)):  # Iterate each sequence
    seq_df = seq.copy()  # Copy sequence DataFrame
    # For each ToF sensor compute pixel aggregates (treat -1 as no return -> NaN)
    for i in range(1, 6):
        pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]  # 64 pixels for this ToF sensor
        tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)  # Replace -1 with NaN
        seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)  # Row mean (per step)
        seq_df[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)  # Row std
        seq_df[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)  # Row min
        seq_df[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)  # Row max
        
    # Missing filling: forward-fill/backward-fill then fill 0
    mat = seq_df[feature_cols].ffill().bfill().fillna(0).values.astype('float32')  # Build the sequence's feature matrix
    all_steps_for_scaler_list.append(mat)  # Collect steps for fitting scaler
    X_list.append(mat)  # Save this sequence matrix
    # y uses composite labels encoded to one-hot later
    y_list.append(label_array[idx])  # Store label index (one-hot later)
    hand_list.append(seq['handedness'].iloc[0])  # Record handedness
    id_list.append(seq_id)  # Record sequence ID
    lens.append(len(mat))  # Record sequence length

# ========== Standardization ==========
print("  Fitting StandardScaler…")  # Fit standardizer
all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)  # Stack all steps into a big matrix
scaler = StandardScaler().fit(all_steps_concatenated)  # Fit StandardScaler (per-column z-score)
joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")  # Persist scaler
del all_steps_for_scaler_list, all_steps_concatenated  # Free memory

print("  Scaling and padding sequences…")  # Scale and pad
X_list = [scaler.transform(x_seq) for x_seq in X_list]  # Standardize each sequence by column

np.save(EXPORT_DIR / "sequence_maxlen.npy", maxlen)  # Save pad length
np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))  # Save feature column names
id_list = np.array(id_list)  # To numpy array
hand_list = np.array(hand_list)  # To numpy array
X_list_all = pad_sequences_torch(X_list, maxlen=maxlen, padding='pre', truncating='pre')  # Pad to fixed length
y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encode labels (composite classes)

# Compose augmenter
augmenter = Augment(p_jitter=0.95, sigma=0.03, scale_range=(0.8, 1.2))

metric_loss_fn = losses.TripletMarginLoss(margin=0.2)  # Triplet loss, margin 0.2
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")  # Hard example miner

EPOCHS = 160  # Total epochs

# Grouping (ensure subjects do not leak across folds)
groups_by_subject = seq_gp['subject'].first()

# oof: all-sensor predictions
oof = np.zeros((X_list_all.shape[0], ), dtype=np.float32)  # Store OOF predictions (class index)

# Collate wrapper (gating/masking)
collate_fn_with_args = partial(mixup_collate_fn, imu_dim=len(imu_cols), masking_prob=MASKING_PROB, )

# Stratified + Grouped K-fold (stratify on binary sequence_type, group by subject)
skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)  # Build stratified grouped K-fold
for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, seq_gp['sequence_type'].first(), groups=groups_by_subject)):
    # The split above: stratify by sequence_type and group by subject; get per-fold train/val indices
    best_h_f1 = 0  # Track best metric for this fold
    train_list= X_list_all[train_idx]  # Train features
    train_y_list= y_list_all[train_idx]  # Train labels
    val_list = X_list_all[val_idx]  # Validation features
    val_y_list= y_list_all[val_idx]  # Validation labels

    # DataLoader
    train_dataset = CMI3Dataset(train_list, train_y_list, maxlen, mode="train", augment=augmenter)  # Train dataset (with augmentation)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_args, num_workers=0, drop_last=True)  # Train DataLoader (custom collate)

    val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")  # Validation dataset (no augmentation)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=False)  # Validation DataLoader

    # Model / optimizer / scheduler
    model = TwoBranchModel(
        maxlen, 
        len(imu_cols), 
        len(tof_cols), 
        len(le.classes_)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR_INIT, weight_decay=WD)  # AdamW optimizer

    # CosineLR + warmup
    nbatch = len(train_loader)  # Batches per epoch
    warmup = 20 * nbatch  # Total warmup steps
    nsteps = EPOCHS * nbatch  # Total training steps
    scheduler = CosineLRScheduler(optimizer, warmup_t=warmup, warmup_lr_init=1e-5, warmup_prefix=True, t_initial=(nsteps - warmup), lr_min=1e-9)  # Cosine schedule with warmup

    train_loss = 0.0  # Accumulated train loss
    train_acc = 0.0  # Placeholder: train metric (H-F1)
    val_loss = 0.0  # Accumulated val loss
    val_acc = 0.0  # Validation metric (H-F1)
    val_best_acc = 0.0  # Unused (kept)
    i_scheduler = 0  # Scheduler step counter
    
    print("▶ Starting training…")  # Start training for this fold
    for epoch in range(EPOCHS):  # Iterate epochs
        model.train()  # Training mode
        train_preds = []  # Store train argmax predictions (for metric approximation)
        train_targets = []  # Store train argmax targets

        for X, y, gate_target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):  
            X, y, gate_target = X.float().to(device), y.to(device), gate_target.to(device)  # Move batch to device and ensure dtypes

            optimizer.zero_grad()  # Zero gradients
            logits, embedding, gate_pred = model(X)  # logits: (B, n_classes), gate_pred: (B, 1)

            # Classification loss: soft CE to one-hot targets (equivalent to KL to soft labels)
            classification_loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()

            # Triplet (hard mining)
            hard_triplets = miner(embedding, y.argmax(dim=1))  # Mine hard triplets (within batch)
            metric_loss = metric_loss_fn(embedding, y.argmax(dim=1), hard_triplets)  # Compute triplet loss

            # Gate loss: predict whether ToF/THM was used (paired with collate's gate_target)
            loss = classification_loss + metric_loss + 0.2 * F.binary_cross_entropy(gate_pred.squeeze(-1), gate_target)  # Total loss (weighted)

            loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer step

            # Record argmax (for H-F1 monitoring)
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())  # Save predicted class index
            train_targets.extend(y.argmax(dim=1).cpu().numpy())  # Save true class index

            scheduler.step(i_scheduler)  # Advance LR scheduler one step
            i_scheduler +=1  # Increment step counter

            train_loss += loss.item()  # Accumulate train loss
            
        # ========== Validation ==========
        model.eval()  # Switch to evaluation mode
        with torch.inference_mode():
            val_preds = []  # Collect validation predictions
            val_targets = []  # Collect validation targets
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                X, y = X.float().to(device), y.to(device)  # Move to device
                logits = model(X)[0]
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())  # Collect predictions
                val_targets.extend(y.argmax(dim=1).cpu().numpy())  # Collect targets
                loss = F.cross_entropy(logits, y)  # Validation cross-entropy (y is one-hot; broadcasting supported)
                val_loss += loss.item()  # Accumulate validation loss

        # Write current fold's validation argmax into OOF (store class indices)
        oof[val_idx] = val_preds  # Update OOF for this fold

        # Map composite labels back to true gesture (extract the gesture field via split('_')[1])
        train_pred_true = [x.split('_')[1] for x in le.classes_[train_preds]]  # Train predictions → gesture
        val_pred_true = [x.split('_')[1] for x in le.classes_[val_preds]]  # Val predictions → gesture
        train_targets_true = [x.split('_')[1] for x in le.classes_[train_targets]]  # Train targets → gesture
        val_targets_true = [x.split('_')[1] for x in le.classes_[val_targets]]  # Val targets → gesture

        # Compute hierarchical H-F1 (approximation of official metric):
        train_acc = CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': train_targets_true}),
            pd.DataFrame({'gesture': train_pred_true}))  # Train H-F1
        val_acc = CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': val_targets_true}),
            pd.DataFrame({'gesture': val_pred_true}))  # Validation H-F1
        train_loss /= len(train_loader)  # Average train loss
        val_loss /= len(val_loader)  # Average validation loss

        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train H-F1: {train_acc:.4f} | Valid H-F1: {val_acc:.4f}")  # Log this epoch
    
        # Track best and save weights
        if val_acc > best_h_f1:  # If current H-F1 is the best
            best_h_f1 = val_acc  # Update best
            torch.save({
            'model_state_dict': model.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'n_classes': len(le.classes_),
            'pad_len': maxlen
            }, EXPORT_DIR / f"gesture_two_branch_fold{fold}.pth")  # Save weights and meta
            print(f"  New best model saved with H-F1: {best_h_f1:.4f}")  # Log save
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1  # No improvement
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")  # Early stopping progress

        if patience_counter >= PATIENCE:  # Patience exceeded
            print("  Early stopping triggered.")  # Early stopping
            break  # End this fold

        print("\n")
      
    print(f"fold: {fold} val_all_acc: {best_h_f1:.4f}")  # Log best H-F1 for this fold
    print("✔ Training done – artefacts saved in", EXPORT_DIR)  # Training done and artifacts path


# ================================
# OOF and IMU-only vs All-modalities comparison
# ================================
val_pred_true = le.classes_[val_preds.argmax(1)]  # Map class indices to composite class names
val_pred_true = [x.split('_')[1] for x in val_pred_true]  # Extract gesture
val_target_true = [x.split('_')[1] for x in le.classes_[val_targets]]  # Extract true gesture

# Rebuild stratified folds (same as above), evaluate saved best weights
groups_by_subject = seq_gp['subject'].first()  # Get grouping basis again

skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)  # Build new stratified grouped K-fold

oof = np.zeros((X_list_all.shape[0], ), dtype=np.float32)  # Store all-sensor OOF
oof_imu = np.zeros((X_list_all.shape[0], ))  # Store IMU-only OOF
for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, seq_gp['sequence_type'].first(), groups=groups_by_subject)):
    _, val_list = X_list_all[train_idx], X_list_all[val_idx]  # Only validation set needed
    _, val_y_list = y_list_all[train_idx], y_list_all[val_idx]  # Validation ground truth

    val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")  # Build validation dataset
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)  # DataLoader
    
    # Load the best weights for this fold
    best_model_path = EXPORT_DIR/ f"gesture_two_branch_fold{fold}.pth"  # Model path
    print(f"Evaluating fold {fold+1} with model {best_model_path}")  # Log evaluation info
    
    eval_model = TwoBranchModel(
        maxlen, 
        len(imu_cols), 
        len(tof_cols), 
        len(le.classes_)
        ).to(device)
    
    eval_model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])  # Load weights
    eval_model.to(device)  # Ensure on device
    eval_model.eval()  # Evaluation mode

    with torch.inference_mode():
        val_preds = []  # IMU-only predictions
        val_targets = []  # Validation ground truth
        val_preds2 = []  # All-sensor predictions
        for X, y in tqdm(val_loader, desc=f"[Val]"):
            # all-sensor
            logits2 = eval_model(X.float().to(device))[0]  # Direct all-modal input
            val_preds2.extend(logits2.argmax(dim=1).cpu().numpy())  # Collect all-modal predictions
        
            # IMU-only: zero out ToF/THM channels
            half = BATCH_SIZE // 2           # Split batch  
            x_front = X[:half]               # Front half keeps all modalities
            x_back  = X[half:].clone()
            x_back[:, :, len(imu_cols):] = 0.0    # Back half zero ToF/THM channels
            X_cat = torch.cat([x_front, x_back], dim=0)  # Concatenate back to one batch
            X_cat, y = X_cat.float().to(device), y.to(device)  # Move to device
            
            logits = eval_model(X_cat)[0]  # Forward
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())  # Collect IMU-only argmax
            val_targets.extend(y.argmax(dim=1).cpu().numpy())  # Collect targets

    oof_imu[val_idx] = val_preds  # Write IMU-only OOF for this fold
    oof[val_idx] = val_preds2  # Write all-sensor OOF for this fold

    val_pred_true = le.classes_[val_preds]  # Map class index to composite name
    val_pred_true = [x.split('_')[1] for x in val_pred_true]  # Extract gesture

    val_pred2_true = [x.split('_')[1] for x in le.classes_[val_preds2]]  # All-sensor gesture
    val_target_true = [x.split('_')[1] for x in le.classes_[val_targets]]  # Ground-truth gesture

    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': val_target_true}),
        pd.DataFrame({'gesture': val_pred_true})
    )  # Compute IMU-only H-F1

    all_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': val_target_true}),
        pd.DataFrame({'gesture': val_pred2_true})
    )  # Compute all-sensor H-F1

    print(f"Fold {fold+1} H-F1 = {h_f1:.4f}")  # Log IMU-only results
    print(f"Fold {fold+1} All H-F1 = {all_h_f1:.4f}")  # Log all-sensor results
    print("\n")