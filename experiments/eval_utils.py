import time
import os

from modal_config import vla_image

with vla_image.imports():
    import imageio
    import numpy as np

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

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