
import torch
import os, json, joblib, numpy as np, pandas as pd
import random
from torch.utils.data import Dataset, DataLoader


def augment_left_handed_sequence(seq_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror left-handed (handedness==0) sequences into right-hand semantics:
    - Swap ToF/THM sensors 3 and 5 (mirror watch layout), keep others the same
    - Negate certain IMU axes (acc_x, rot_y, rot_z) to approximate mirroring
    """
    if seq_df['handedness'].iloc[0] == 0:
        seq_df = seq_df.copy()

        # Collect columns for tof_3/thm_3 and tof_5/thm_5
        cols_3 = [c for c in seq_df.columns if any(p in c for p in ['tof_3', 'thm_3'])]
        cols_5 = [c for c in seq_df.columns if any(p in c for p in ['tof_5', 'thm_5'])]
        cols_3.sort(); cols_5.sort()
        # Swap 3 and 5
        if len(cols_3) == len(cols_5) and len(cols_3) > 0:
            temp_data_3 = seq_df[cols_3].values
            seq_df[cols_3] = seq_df[cols_5].values
            seq_df[cols_5] = temp_data_3
        
        negate_cols = ['acc_x', 'rot_y', 'rot_z']
        existing_negate_cols = [col for col in negate_cols if col in seq_df.columns]
        if existing_negate_cols:
            seq_df[existing_negate_cols] *= -1
    return seq_df


def mixup_collate_fn(batch, imu_dim, masking_prob=0.0):
    """Custom collate:
    - With probability masking_prob, zero out ToF/THM channels and output gate_target (to supervise the gate head)
    Returns: X_batch (augmented), y_batch (same distribution mix), gate_target ([0,1])
    """
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    
    batch_size, seq_len, _ = X_batch.shape

    gate_target = torch.ones(batch_size, dtype=torch.float32)
    if masking_prob > 0:
        mask = torch.rand(batch_size) < masking_prob
        X_batch[mask, :, imu_dim:] = 0
        gate_target[mask] = 0.0

    return X_batch, y_batch, gate_target


class Augment:
    def __init__(self, p_jitter, sigma, scale_range):      
        self.p_jitter = p_jitter
        self.sigma = sigma
        self.scale_min, self.scale_max = scale_range
        
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min, self.scale_max, size=(1, x.shape[1]))
        return (x + noise) * scale
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)
        return x


class CMI3Dataset(Dataset):
    """CMI-DBSD sequence-level dataset:
    - X_list: list[np.ndarray], each is (T_i, D) time-series features
    - y_list: list[np.ndarray], each is one-hot (C,) or index
    - augment: optional augmenter, used only in training mode
    - Returns: (maxlen, D) tensor after padding + target
    """
    def __init__(self,
                 X_list,
                 y_list,
                 maxlen,
                 mode="train",
                 augment=None):
        
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.maxlen = maxlen
        self.augment = augment

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):
        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        if self.mode == "train" and self.augment is not None:
            X = self.augment(X)

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        return torch.from_numpy(X.copy()).float(), torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X_list)