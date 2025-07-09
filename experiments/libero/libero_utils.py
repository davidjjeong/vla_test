import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modal_config import app, data_vol, vla_image

with vla_image.imports():
    import tensorflow as tf
    import numpy as np
    from huggingface_hub import snapshot_download

    from vla_test.data.libero.libero import get_libero_path
    from vla_test.data.libero.libero.envs import OffScreenRenderEnv

# Download LIBERO datasets (for one-time use)
@app.function(
    image=vla_image,
    timeout=3600,    # Default timeout of Modal function is 300s, hence needed to extend
    volumes={"/root/vla_test/data/libero/datasets": data_vol}
)
def download_data(repo_id: str = "yifengzhu-hf/LIBERO-datasets", local_dir: str = "/root/vla_test/data/libero/datasets"):
    folder_path = snapshot_download(
        repo_id=repo_id, 
        repo_type="dataset", 
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN")
    )

def get_libero_env(task, resolution: int = 256):
    task_description = task.language
    task_bddle_dir = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddle_dir, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description

def get_libero_dummy_action():
    """Get dummy action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_img_resize_dim(model_id: str = "nora"):
    if model_id == "nora":
        resize_dim = 224    # nora expects image of size 224 by 224
    else:
        raise ValueError(f"Unexpected `model_id`: {model_id} given.")
    return resize_dim

def resize_img(img, resize_dim):
    """
    Takes a numpy array corresponding to an image and returns resized image as a numpy array.
    """
    assert isinstance(resize_dim, tuple)    # Ensure that resize_dim is a tuple of height and width
    # Resize img to size expected by the model
    img = tf.image.encode_jpeg(img)         # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_dim, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

def get_preprocessed_img(obs, resize_dim):
    assert isinstance(resize_dim, int) or isinstance(resize_dim, tuple)
    if isinstance(resize_dim, int):
        resize_dim = (resize_dim, resize_dim)
    img = obs["agentview_image"]
    img = resize_img(img, resize_dim)
    return img