import modal
from pathlib import Path

# Set environment variables
LOCAL_PROJECT_DIR = Path(__file__).parent
LIBERO_PATH = "/root/vla_test/data"
NORA_PATH = "/root/vla_test/models/nora"

# Define base Modal image
vla_image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libegl1-mesa-dev", # For EGL support
        "libosmesa6-dev",   # For OSMesa (off-screen rendering)
        "libglfw3",
        "libglfw3-dev",
        "ffmpeg",
        "xvfb",             # For virtual display, if needed
        "unzip",            # For extracting datasets
        "cmake",            # Often required for compiling packages
        "g++"
    )
    .pip_install(
        "transformers==4.53.0",         # Recent stable version
        "huggingface_hub==0.33.2",      # Recent stable version
        "torch==2.4.0",                 # Version used by NORA
        "requests",
        "urllib3",
        "accelerate",
        "tokenizers",
        "filelock",
        "packaging",
        "pyyaml",
        "tqdm",
        "datasets",
        "Pillow",
        "numpy",
        "scipy",
        "matplotlib",
        "future",
        "thop",
        "einops",
        "robomimic",
        "opencv-python",
        "wandb",
        "hydra-core",
        "imageio[ffmpeg]",              
        "robosuite==1.4.1",
        "bddl",                        
        "easydict",                     
        "cloudpickle",                  
        "gym",
        "tensorflow",
        "qwen_vl_utils",
        "torchvision"                       
    )
    .run_commands(
        "echo 'Cloning LIBERO repository into image...'",
        f"git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git {LIBERO_PATH}",
        f"cd {LIBERO_PATH} && pip install -e .",
        "echo 'Successfully cloned LIBERO repository into image.'",
        "echo 'Cloning NORA repository into image...'",
        f"git clone https://github.com/declare-lab/nora.git {NORA_PATH}",
        "echo 'Successfully cloned NORA repository into image.'"
    )
    .add_local_dir(LOCAL_PROJECT_DIR, remote_path="/root")
)

# Define Modal volume
# The volume will be used to permanently store LIBERO data and VLA models.
data_vol = modal.Volume.from_name("data-cache", create_if_missing=True)

# Define Modal app
app = modal.App(
    name="vla-model-summary",
    image = vla_image,
    secrets=[modal.Secret.from_name("vla-eval-secret")]
)