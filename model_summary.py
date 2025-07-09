import modal
import os

from modal_config import vla_image, data_vol, app
from experiments.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_img_resize_dim,
    get_preprocessed_img
)

# Import necessary packages
with vla_image.imports():
    import tqdm

    from vla_test.data.libero.libero import benchmark
    from vla_test.models.nora.inference.nora import Nora

@app.cls(
    image=vla_image,
    volumes={
        "/root/vla_test/data/libero/datasets": data_vol,
    },
    #gpu="T4",
)
class ModelSummary():
    """
    Modal class to load and evaluate Vision-Language-Action (VLA) models.
    Instance of this class will persist in the cloud, allowing the model to
    be loaded once and be reused again.
    """

    model_id: str = modal.parameter(default="nora")
    eval_data_id: str = modal.parameter(default="libero_spatial")
    num_steps_wait: int = modal.parameter(default=10)
    num_trials_per_task: int = modal.parameter(default=50)

    @modal.enter()
    def init_summary(self):
        # Initialize variables
        self.model = None                                           # stores the loaded model object
        self.eval_summary = {}                                      # stores the evaluation summary
        self.hf_token = os.environ.get("HF_TOKEN")                  # token to access models and datasets on HF

        # --- Load Model ---
        if self.model_id == "nora":
            try:
                print(f"Loading VLA model '{self.model_id}'...")
                self.model = Nora(device = 'cuda')
                print(f"Successfully loaded '{self.model_id}'!")
            except Exception as e:
                raise RuntimeError(f"Failed to load '{self.model_id}': {e}")
        else:
            raise ValueError(f"Model '{self.model_id}' unsupported. Please check model availability.")

    @modal.method()
    def eval_model_on_libero(self):
        """
        Evaluate specified LIBERO task suite on the loaded model.
        """
        if self.model is None:
            raise RuntimeError(f"Cannot load model for evaluation. Check if '{self.model_id}' has been inputted correctly, \
                                or the pre-trained model exists in cache.")
        
        """
        Evaluation code adapted from [OpenVLA]: https://github.com/openvla/openvla
        """
        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.eval_data_id]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Initialized '{self.eval_data_id}' data of for evaluation.\nNumber of tasks: {num_tasks_in_suite}")

        # Get expected image dimensions for the loaded model
        resize_dim = get_img_resize_dim(self.model_id.split("/")[1])

        # Start evaluation
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, resolution=256)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(self.num_trials_per_task)):
                print(f"\nTask: {task_description}")

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                if self.eval_data_id == "libero_spatial":
                    max_steps = 220     # longest training demo has 193 steps
                elif self.eval_data_id == "libero_object":
                    max_steps = 280     # longest training demo has 254 steps
                elif self.eval_data_id == "libero_goal":
                    max_steps = 300     # longest training demo has 270 steps
                elif self.eval_data_id == "libero_10":
                    max_steps = 520     # longest training demo has 505 steps
                elif self.eval_data_id == "libero_90":
                    max_steps = 400     # longest training demo has 373 steps
                
                print(f"Starting episode {task_episodes+1}...")
                while t < max_steps + self.num_steps_wait:
                    try:
                        # Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall in the environment.
                        if t < self.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        # Get preprocessed image
                        img = get_preprocessed_img(obs, resize_dim)

                        t += 1
                    except Exception as e:
                        raise RuntimeError(f"Caught exception: {e}")

@app.function(image=vla_image)
def main():
    noraSummary = ModelSummary()
    noraSummary.eval_model_on_libero.remote()