import modal
import os

from modal_config import vla_image, models_vol, data_vol, app
from experiments.libero.libero_utils import get_libero_env

# Import necessary packages
with vla_image.imports():
    import tqdm
    from datasets import load_dataset
    from transformers import AutoModel

    from vla_test.data.libero.libero import (
        benchmark
    )

@app.cls(
    image=vla_image,
    volumes={
        "/root/vla_test/models": models_vol,
        "/root/vla_test/data/libero/datasets": data_vol,
    },
    gpu="T4",
)
class ModelSummary():
    """
    Modal class to load and evaluate Vision-Language-Action (VLA) models.
    Instance of this class will persist in the cloud, allowing the model to
    be loaded once and be reused again.
    """
    model_id: str = modal.parameter(default="declare-lab/nora")
    eval_data_id: str = modal.parameter(default="libero_spatial")

    @modal.enter()
    def init_report(self):
        # Initialize variables
        self.model = None                                       # stores the loaded model object
        self.eval_summary = {}                                  # stores the evaluation summary
        self.eval_data_dir = "lerobot/" + self.eval_data_id     # path to specified dataset on Hugging Face(HF)
        self.hf_token = os.environ.get("HF_TOKEN")              # token to access models and datasets on HF

        # --- Load Model ---
        try:
            print(f"Loading VLA model '{self.model_id}'...")

            self.model = AutoModel.from_pretrained(
                self.model_id, 
                torch_dtype="auto", 
                device_map="auto",
                cache_dir="/root/vla_test/models",
                token=self.hf_token,
                trust_remote_code=True
            )
            print(f"Successfully loaded '{self.model_id}'!")
        except Exception as e:
            raise RuntimeError(f"Failed to load '{self.model_id}' from Hugging Face: {e}")

    @modal.method()
    def load_eval_data(self):
        """
        Load eval_data from HF
        
        For now, we focus on the LeRobot adaptation of LIBERO dataset.
        Later, other data can be added in the future for further eval.
        """
        try:
            print(f"Loading '{self.eval_data_id}' from Hugging Face...")
            self.eval_data = load_dataset(self.eval_data_dir, split="train", trust_remote_code=True, token=self.hf_token)
            print(f"Successfully loaded '{self.eval_data_id}'!")
        except Exception as e:
            raise RuntimeError(f"Error loading '{self.eval_data_id}' from Hugging Face: {e}")

    @modal.method()
    def eval_model(self):
        """
        Evaluate loaded eval_data on the loaded model.
        """
        
        if self.model is None:
            raise RuntimeError(f"Cannot load model for evaluation. Check if '{self.model_id}' has been inputted correctly, \
                                or the pre-trained model exists on Hugging Face.")
        """
        if self.eval_data is None:
            raise RuntimeError(f"Cannot load data for evaluation. Check if '{self.eval_data_id}' has been inputted correctly, \
                                or the dataset exists on Hugging Face.")
        """
        
        """
        Evaluation code adapted from [OpenVLA]: https://github.com/openvla/openvla
        """
        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.eval_data_id]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Initialized '{self.eval_data_id}' data of for evaluation.\nNumber of tasks: {num_tasks_in_suite}")

        # Start evaluation
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, resolution=256)

@app.function(image=vla_image)
def main():
    noraSummary = ModelSummary()
    noraSummary.eval_model.remote()