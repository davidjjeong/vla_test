import modal
from pathlib import Path

# Set environment variables
LOCAL_PROJECT_DIR = Path(__file__).parent
LIBERO_PATH = "/root/vla_test/data"
NORA_PATH = "/root/vla_test/models/nora"
GR00T_PATH = "/root/vla_test/models/gr00t"

#############################################################################

"""
Modal configuration for NORA evaluation on LIBERO.
"""

# Define base Modal image using an official TensorFlow GPU image
# This image comes with Python, CUDA Toolkit (including libdevice), cuDNN,
# and TensorFlow pre-installed and configured for GPU.
tensorflow_gpu_base_image = modal.Image.from_registry(
    "tensorflow/tensorflow:2.16.1-gpu",
    add_python="3.10"
)

# Now, layer additional installations on top of this GPU-ready base image.
nora_image = (
    tensorflow_gpu_base_image
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
        "g++",
        "python3-dev",      # For building Python C extensions
        "libudev-dev",      # Required by evdev for device interaction
        "clang",            # Install clang as evdev explicitly tries to use it
        "build-essential",  # Ensures core compilation tools are present
    )
    .pip_install(
        "tensorflow",
        "transformers==4.50.0",         # Version used by NORA
        "huggingface_hub",      
        "torch==2.4.0",                 # Version used by NORA
        "requests",
        "urllib3",
        "accelerate==1.5.2",            # Version used by NORA
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
        "qwen_vl_utils",
        "torchvision==0.19.0",          # Version used by NORA                  
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

# Define Modal app
nora_app = modal.App(
    name="nora-eval-summary",
    image = nora_image,
    secrets=[modal.Secret.from_name("vla-eval-secret")]
)

#############################################################################

"""
Modal configuration for GR00T N1.5 evaluation on LIBERO.
"""

# Define Modal image
gr00t_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(           # apt_install copied from nora_image config
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
        "g++",
        "python3-dev",      # For building Python C extensions
        "libudev-dev",      # Required by evdev for device interaction
        "clang",            # Install clang as evdev explicitly tries to use it
        "build-essential",  # Ensures core compilation tools are present
    )
    .pip_install(
        "imageio[ffmpeg]",
        "robosuite == 1.4.1",
        "bddl",
        "easydict",
        "cloudpickle",
        "gym",
        "ninja"             # Speeds up the process of build wheel for flash-attn
    )
    .run_commands(
        "echo 'Cloning LIBERO repository into image...'",
        f"git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git {LIBERO_PATH}",
        "pip install --upgrade setuptools",
        f"cd {LIBERO_PATH} && pip install -e .",
        "echo 'Successfully cloned LIBERO repository into image.'",
        "echo 'Cloning GR00T repository into image...'",
        f"git clone https://github.com/davidjjeong/Isaac-GR00T.git {GR00T_PATH}",
        f"cd {GR00T_PATH} && pip install -e . && MAX_JOBS=4 pip install flash-attn==2.7.1.post4 --no-build-isolation",
        "echo 'Successfully cloned GR00T repository into image.'"
    )
)

# Define Modal app
gr00t_app = modal.App(
    name="gr00t-eval-summary",
    image = gr00t_image,
    secrets=[modal.Secret.from_name("vla-eval-secret")]
)

#############################################################################

# Define Modal volumes
# data_vol: permanent storage of evaluation data (LIBERO)
# rollouts_vol: permanent storage of rollout videos from evaluation
data_vol = modal.Volume.from_name("data-cache", create_if_missing=True)
rollouts_vol = modal.Volume.from_name("rollouts-cache", create_if_missing=True)
eval_summary_vol = modal.Volume.from_name("eval-summary-cache", create_if_missing=True)