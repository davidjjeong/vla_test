import time
import os
import random
import math

from modal_config import nora_image, gr00t_image

with nora_image.imports() or gr00t_image.imports():
    import imageio
    import numpy as np
    import torch

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

def set_seed_config(seed: int = 7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_rollout_video(rollout_images, episode_idx, task_description, success, fps: int = 30):
    """
    Saves a numpy array (np.ndarray) of images as a video at the specified directory.
    """
    video_dir = f"/root/vla_test/rollouts/{DATE}"
    os.makedirs(video_dir, exist_ok=True)   # Make directory if video_dir does not exist
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    video_dir = f"{video_dir}/{DATE_TIME}--episode={episode_idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(video_dir, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

    print(f"Video saved successfully to {video_dir}.")

def normalize_gripper_action(action, binarize=True):
    """
    Adapted from [OpenVLA].

    Changes gripper action (last dimension of action vector) from [0, 1] to [-1, +1]
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        action[..., -1] = 1.0 if action[...,-1] > 0.0 else -1.0
    
    return action

def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action 

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den