import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modal_config import app, data_vol, vla_image

with vla_image.imports():
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