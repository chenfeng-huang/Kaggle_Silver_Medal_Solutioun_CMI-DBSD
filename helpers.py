import os, numpy as np, pandas as pd
import random
from scipy.spatial.transform import Rotation as R
import torch


# %% ----------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set global random seeds across libraries to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% ----------------------------------------------------------------
def remove_gravity_from_acc(acc_data, rot_data):
    """
    Remove gravity from IMU acceleration using quaternion orientation to get linear acceleration.

    1) Why linear acceleration? How is it different from raw acc_[x/y/z]?
    - Accelerometers read ~9.81 m/s² even at rest due to gravity.
    - We care about motion-induced acceleration (raising hand, scratching, pinching). Linear acceleration is raw acc minus gravity.

    2) Why need quaternions rot_[x/y/z/w] to subtract gravity from acc_[x/y/z]?
    - Gravity is fixed in world frame [0,0,9.81], but sensor axes rotate with the device.
    - Use the current orientation (quaternions) to rotate the world gravity into the sensor frame, then acc - g_sensor.

    Parameters
    -----
    acc_data : pd.DataFrame or ndarray with acc_x, acc_y, acc_z
    rot_data : pd.DataFrame or ndarray with rot_x, rot_y, rot_z, rot_w (quaternion)
    Returns
    -----
    linear_accel : ndarray with the same shape as acc_data (three-axis linear acceleration)
    Notes
    -----
    • gravity_world = [0,0,9.81]
    • Use scipy.spatial.transform.Rotation to rotate gravity into the sensor frame; acc - g_sensor -> linear acc
    • Fallback to raw acceleration when quaternion is invalid/NaN
    """
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
            
    return linear_accel


def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    """Approximate angular velocity (rad/s) from a quaternion sequence.

    1) Angular velocity ≈ (small change in pose) / (small time)
    - v = Δx / Δt

    2) Why angular velocity?
    - It describes how the wrist is rotating: speed and direction; dynamic characteristics vary by gesture.

    Parameters
    -----
    rot_data : DataFrame/ndarray, quaternions [x,y,z,w]
    time_delta : float, time step between samples (Hz=200 -> dt=1/200)
    Returns
    -----
    angular_vel : ndarray of shape (T, 3)
    Notes
    -----
    Use relative rotation ΔR = R_t^{-1} * R_{t+dt}, take its rotation vector and divide by dt.
    """
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
            
    return angular_vel


def calculate_angular_distance(rot_data):
    """Compute angular distance Δθ (radians) between adjacent quaternions.
    1) Angular distance is the minimal rotation angle from A to B, in [0, π].
    Returns a single-channel sequence capturing pose-change intensity.
    """
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass
            
    return angular_dist


def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """
    Pad/truncate each sequence to maxlen and return a np.float32 array.
    """
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)

