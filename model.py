import torch
import torch.nn as nn
import torch.nn.functional as F

class ImuFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        k = 15  # 1D kernel size used as the length of a low-pass filter
        
        self.lpf_acc   = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)  # Low-pass filter for acceleration to suppress high-frequency noise
        nn.init.kaiming_normal_(self.lpf_acc.weight, mode='fan_out') # Kaiming init

        self.lpf_gyro  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)  # Low-pass filter for gyroscope to suppress high-frequency noise
        nn.init.kaiming_normal_(self.lpf_gyro.weight, mode='fan_out') # Kaiming init

    def forward(self, imu):
        '''
        imu: input tensor shape (B, C, T), B=batch size, C=#IMU channels, T=time length
        C layout:
         - 0:3=acc_xyz, 3=rot_w, 4:7≈rot_xyz, 7:10=linear_acc_xyz, 10:13=angular_vel_xyz, 13:14=angular_distance.
        '''
 
        acc  = imu[:, 0:3, :]                 # Acceleration (acc_x, acc_y, acc_z), (B,3,T)
        gyro = imu[:, 4:7, :]                 # Gyroscope (gyro_x, gyro_y, gyro_z), (B,3,T); channel 3 typically rot_w (quat w)
        linear_acc      = imu[:, 7:10, :]     # Linear acceleration (gravity removed), (B,3,T)
        angular_vel     = imu[:, 10:13, :]    # Angular velocity, (B,3,T)
        angular_distance = imu[:, 13:14, :]   # Angular distance (scalar), (B,1,T)

        linear_acc_mag = torch.norm(linear_acc, dim=1, keepdim=True)  # Magnitude of linear acceleration, (B,1,T)
        linear_acc_mag_jerk = F.pad(linear_acc_mag[:, :, 1:] - linear_acc_mag[:, :, :-1], (1,0), 'replicate')  # First diff of linear acc magnitude, (B,1,T)

        angular_vel_mag = torch.norm(angular_vel, dim=1, keepdim=True)  # Magnitude of angular velocity, (B,1,T)
        angular_vel_mag_jerk = F.pad(angular_vel_mag[:, :, 1:] - angular_vel_mag[:, :, :-1], (1,0), 'replicate')  # First diff of ang vel magnitude, (B,1,T)

        rot_angle = 2 * torch.acos(imu[:, 3, :].clamp(-1.0, 1.0)).unsqueeze(1)  # Absolute rotation angle, (B,1,T)
        rot_angle_vel = F.pad(rot_angle[:, :, 1:] - rot_angle[:, :, :-1], (1,0), 'replicate')  # First diff of abs rot angle, (B,1,T)

        # 1) Magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # Acc magnitude (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)          # Gyro magnitude (B,1,T)

        # 2) First-order difference (jerk/delta)
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))          # Acc first diff (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))  # Gyro first diff (B,3,T)

        # 3) Power (squared intensity)
        acc_pow  = acc ** 2                                       # Acc squared (B,3,T)
        gyro_pow = gyro ** 2                                      # Gyro squared (B,3,T)

        # 4) LPF/HPF decomposition (smooth vs fast-changing components)
        acc_lpf  = self.lpf_acc(acc)                              # Acc low-pass (B,3,T)
        acc_hpf  = acc - acc_lpf                                  # Acc high-pass (B,3,T)
        gyro_lpf = self.lpf_gyro(gyro)                            # Gyro low-pass (B,3,T)
        gyro_hpf = gyro - gyro_lpf                                # Gyro high-pass (B,3,T)

        # Aggregate acceleration-related features (raw, magnitude, jerk, power, filters, linear acc family), 21 channels
        acc_features = [
            acc, acc_mag,
            jerk, acc_pow,
            acc_lpf, acc_hpf,
            linear_acc, linear_acc_mag, linear_acc_mag_jerk,
        ]

        # Aggregate angle/rotation-related features (raw, magnitude, diffs, power, filters, angular velocity family, angular distance, quat-derived angles), 24 channels
        gyro_features = [
            gyro, gyro_mag,
            gyro_delta, gyro_pow,
            gyro_lpf, gyro_hpf,
            angular_vel, angular_vel_mag, angular_vel_mag_jerk, angular_distance,
            rot_angle, rot_angle_vel,
        ]

        features = acc_features + gyro_features  # Concatenate feature lists
        return torch.cat(features, dim=1)  # Concatenate along channels -> (B, D_imu_features, T)

class SEBlock(nn.Module):
    '''Channel Attention: adaptively emphasize useful channels and suppress less useful ones per sample.'''
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Squeeze over time dimension -> (B,C,1)
        self.excitation = nn.Sequential(        # Excitation MLP: C -> C/reduction -> C, sigmoid weights
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()               # Parse batch and channels
        y = self.squeeze(x).view(b, c)   # Pool time and flatten -> (B,C)
        y = self.excitation(y).view(b, c, 1)  # Channel weights (B,C,1)
        return x * y.expand_as(x)        # Channel-wise weighting


class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        
        # Conv block 1: Conv1d + BN
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)  # Length-preserving 1D conv
        self.bn1 = nn.BatchNorm1d(out_channels)  # BatchNorm for training stability
        
        # Conv block 2: Conv1d + BN
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)  # Second conv
        self.bn2 = nn.BatchNorm1d(out_channels)  # BatchNorm
        
        # SE (Squeeze-and-Excitation) channel attention
        self.se = SEBlock(out_channels)
        
        # Residual shortcut (1x1 conv if channel mismatch)
        self.shortcut = nn.Sequential()  # Identity by default
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),  # 1x1 conv to align channels
                nn.BatchNorm1d(out_channels)                           # BN
            )
        
        self.pool = nn.MaxPool1d(pool_size)  # Temporal max pooling to expand receptive field
        self.dropout = nn.Dropout(dropout)   # Dropout to reduce overfitting
        
    def forward(self, x):
        shortcut = self.shortcut(x)  # Shortcut branch (identity or 1x1 conv)
        
        # Conv block 1: Conv -> BN -> SiLU
        out = F.silu(self.bn1(self.conv1(x)))
        # Conv block 2: Conv -> BN (activation later)
        out = self.bn2(self.conv2(out))
        
        # SE channel attention
        out = self.se(out)
        
        # Residual addition
        out += shortcut
        out = F.silu(out)  # Activation after residual fusion
        
        # Pooling and Dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out  # Shape: (B, out_channels, T/pool_size)


class TwoBranchModel(nn.Module):
    '''
    Parameters
    ----
    pad_len : int
        Fixed sequence length L after preprocessing (pre-padding/pre-truncating). Must match input T.

    imu_dim_raw : int
        Number of IMU channels in the input tensor D_imu_raw (used to split x[:, :, :imu_dim_raw]).
        Should match the count of IMU feature columns used during training (e.g., acc_xyz, rot_w/xyz,
        linear_acc_xyz, angular_vel_xyz, angular_distance, in the predefined order).

    tof_dim : int
        Number of ToF/THM channels in the input tensor D_tof (used to split x[:, :, imu_dim_raw:]).
        Should match thm_* originals plus tof_* aggregated stats (per-sensor 64 pixels' mean/std/min/max).

    n_classes : int
        Number of classes (composite labels during training). Forward logits shape is (B, n_classes).

    dropouts : List[float], optional (length must be 7; default [0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3])
        Dropout probabilities mapped to modules in order:
        - dropouts[0] → IMU-acceleration branch block 1 (imu_block11)
        - dropouts[1] → IMU-acceleration branch block 2 (imu_block12)
        - dropouts[2] → ToF/THM branch block 1 (tof_conv1)
        - dropouts[3] → ToF/THM branch block 2 (tof_conv2)
        - dropouts[4] → Post-fusion CNN backbone stage 1 (cnn_backbone1)
        - dropouts[5] → Post-fusion CNN backbone stage 2 (cnn_backbone2) and dropout after dense1 (drop1)
        - dropouts[6] → Dropout after dense2 (drop2)
    '''
    def __init__(self, 
                 pad_len, 
                 imu_dim_raw, 
                 tof_dim, 
                 n_classes, 
                 dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3], 
                 ):
        
        super().__init__()

        self.imu_fe = ImuFeatureExtractor()  # IMU feature extractor (time/magnitude/filter features)

        self.imu_dim = 45       # Effective IMU feature channels
        self.tof_dim = tof_dim       # ToF/Thermopile channels
        self.fir_nchan = imu_dim_raw # Raw IMU channels (for splitting x)
        
        self.acc_dim, self.rot_dim = 21, 24  # Split engineered IMU features: first 21 acc-related, last 24 rotation-related
        self.imu_block11 = ResidualSECNNBlock(self.acc_dim, 64, 3, 1, dropout=dropouts[0])  # Acc branch: small kernel extracts local patterns
        self.imu_block12 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1])           # Acc branch: larger kernel expands receptive field

        self.imu_block21 = ResidualSECNNBlock(self.rot_dim, 64, 3, 1, dropout=dropouts[0])  # Rotation branch: small kernel extracts local patterns
        self.imu_block22 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1])           # Rotation branch: larger kernel expands receptive field

        self.tof_conv1   = ResidualSECNNBlock(tof_dim, 64, 3, 1, dropout=dropouts[2])  # ToF/Thermopile branch
        self.tof_conv2   = ResidualSECNNBlock(64, 128, 3, 1, dropout=dropouts[3])      # ToF/Thermopile branch

        # Gate: adaptively scale the ToF/Thermopile branch by global temporal context
        self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive avg pool to reduce time dimension to 1
        self.dense1_gate = nn.Linear(pad_len, 16)  # Time length pad_len -> 16 (input from pooled (B,T))
        self.dense2_gate = nn.Linear(16, 1)        # Scalar gate value (B,1)
        
        # Post-fusion channels: IMU 128+128=256, ToF 128 → total 384
        merged_channels = 384  # Total concatenated channels
        self.cnn_backbone1 = nn.Sequential(                           # Backbone 1: expand receptive field and mix fused channels
            nn.Conv1d(merged_channels, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropouts[4])
        )
        self.cnn_backbone2 = nn.Sequential(                           # Backbone 2: extract higher-level temporal features
            nn.Conv1d(256, 512, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropouts[5])
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Aggregate time to 1 → (B,512,1)

        # Classification head: two FC layers + BN + SiLU + Dropout, then final classifier
        cnn_out_dim = 512                                 # Channels from backbone
        self.dense1 = nn.Linear(cnn_out_dim, 256, bias=False)  # FC1: 512 -> 256
        self.bn_dense1 = nn.BatchNorm1d(256)                   # BN
        self.drop1 = nn.Dropout(dropouts[5])                   # Dropout
        
        self.dense2 = nn.Linear(256, 128, bias=False)     # FC2: 256 -> 128 (sequence embedding)
        self.bn_dense2 = nn.BatchNorm1d(128)              # BN
        self.drop2 = nn.Dropout(dropouts[6])              # Dropout
        
        self.classifier = nn.Linear(128, n_classes)       # Final classifier: logits over classes
        
    def forward(self, x):
        # x shape assumed (B, T, D_all), D_all = imu_dim_raw + tof_dim
        # Split by raw IMU channels: first self.fir_nchan for IMU, rest for ToF/Thermopile
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # IMU part -> (B, D_imu_raw, T)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # ToF/Thermopile part -> (B, D_tof, T)

        imu  = self.imu_fe(imu)  # IMU feature engineering: (B, D_imu, T)
        
        # Split IMU features into acc-related (first 21) and rotation-related (last 24)
        acc = imu[:, :self.acc_dim, :]   # Acc family (B,21,T)
        rot = imu[:, self.acc_dim:, :]   # Rotation/angle family (B,24,T)

        x11 = self.imu_block11(acc)      # Acc branch: residual SE conv block 1 -> (B,64,T)
        x11 = self.imu_block12(x11)      # Acc branch: residual SE conv block 2 -> (B,128,T)
        
        x12 = self.imu_block21(rot)      # Rotation branch: residual SE conv block 1 -> (B,64,T)
        x12 = self.imu_block22(x12)      # Rotation branch: residual SE conv block 2 -> (B,128,T)
        x1 = torch.cat([x11, x12], dim=1)  # Concatenate along channels; IMU branch total output (B,256,T)

        # ToF/Thermopile branch (note: in hidden test, half samples may have NaN/constant ToF/THM; gate suppresses noise)
        # v2: two residual SE conv layers
        x2 = self.tof_conv1(tof)         # ToF branch block 1 -> (B,64,T)
        x2 = self.tof_conv2(x2)          # ToF branch block 2 -> (B,128,T)

        # Gate the ToF branch by a learned scalar weight based on global temporal profile
        gate_input = self.pool(tof.transpose(1, 2)).squeeze(-1)  # (B,D_tof,T) -> (B,T,D_tof) -> avg over D_tof -> (B,T)
        gate_input = F.silu(self.dense1_gate(gate_input))        # Linear (pad_len->16) + SiLU
        
        gate = torch.sigmoid(self.dense2_gate(gate_input)) # -> (B, 1)
        x2 = x2 * gate.unsqueeze(-1)                        # Broadcast to time and scale ToF branch
        
        merged = torch.cat([x1, x2], dim=1) # Fuse IMU and ToF branches, shape (B, 256+128, T)
        
        # Backbone CNN extracts higher-level temporal features
        cnn_out = self.cnn_backbone1(merged) # Backbone 1 -> (B,256,T)
        cnn_out = self.cnn_backbone2(cnn_out) # Backbone 2 -> (B,512,T)

        # Global pooling to get fixed-length representation
        pooled = self.global_pool(cnn_out) # Adaptive avg pool -> (B,512,1)
        pooled_flat = torch.flatten(pooled, 1) # Flatten -> (B,512)
        
        # Classification head: two FC + BN + SiLU + Dropout
        x = F.silu(self.bn_dense1(self.dense1(pooled_flat)))  # 512 -> 256
        x = self.drop1(x)                                     # Dropout
        x = F.silu(self.bn_dense2(self.dense2(x)))            # 256 -> 128
        x = self.drop2(x)                                     # Dropout
        
        logits = self.classifier(x)                           # Linear classifier to n_classes
        return logits, x, gate                                # Return: logits, 128-dim embedding, ToF gate

