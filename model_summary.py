import modal
import os
import json

from modal_config import vla_image, data_vol, rollouts_vol, eval_summary_vol, app
from experiments.eval_utils import (
    set_seed_config,
    save_rollout_video,
    normalize_gripper_action,
    invert_gripper_action,
    DATE,
    DATE_TIME
)
from experiments.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_img_resize_dim,
    get_preprocessed_img
)

# Import necessary packages
with vla_image.imports():
    import tqdm
    import numpy as np
    from PIL import Image

    from vla_test.data.libero.libero import benchmark
    from vla_test.models.nora.inference.nora import Nora

@app.cls(
    image=vla_image,
    volumes={
        "/root/vla_test/data/libero/datasets": data_vol,
        "/root/vla_test/rollouts": rollouts_vol,
        "/root/vla_test/eval_summary": eval_summary_vol,
    },
    gpu="A100",
    timeout=60*60*24,   # 24hr timeout
    retries=3
)
class ModelSummary():
    """
    Modal class to load and evaluate Vision-Language-Action (VLA) models.
    Instance of this class will persist in the cloud, allowing the model to
    be loaded once and be reused again.
    """

    model_id: str = modal.parameter(default="nora")
    finetune_ok: bool = modal.parameter(default=False)
    eval_data_id: str = modal.parameter(default="libero_spatial")
    num_steps_wait: int = modal.parameter(default=10)
    num_trials_per_task: int = modal.parameter(default=10)          # set to 10 for pre-trained model without fine-tuning

    @modal.enter()
    def init_summary(self):
        # Initialize variables
        self.model = None                                           # stores the loaded model object
        self.eval_summary = {}                                      # stores the evaluation summary
        self.hf_token = os.environ.get("HF_TOKEN")                  # token to access models and datasets on HF

        # --- Load Model ---
        if self.model_id == "nora":
            if self.finetune_ok:
                repo_id = 'declare-lab/' + self.model_id + '-finetuned-' + self.eval_data_id.replace("_", "-")
            else:
                repo_id = 'declare-lab/' + self.model_id

            try:
                print(f"Loading VLA model '{self.model_id}'...")
                self.model = Nora(repo_id, device = 'cuda')
                print(f"Successfully loaded '{self.model_id}'!")
            except Exception as e:
                raise RuntimeError(f"Failed to load '{self.model_id}': {e}")
        else:
            raise ValueError(f"Model '{self.model_id}' unsupported. Please check model availability.")
    
    @modal.method()
    def model_libero_inference(self, image, instruction: str, unnorm_key: str):
        """
        Query model to generate robot action. Return the actions outputted by the model.
        """
        input_image = Image.fromarray(image)
        input_image = input_image.convert("RGB")

        if self.model_id == "nora":
            # ----------------------------------------------------------------------------------------------------
            # Update norm_stats to include norm_stats calculated for LIBERO in the fine-tuned OpenVLA model.
            # ----------------------------------------------------------------------------------------------------
            if unnorm_key not in self.model.norm_stats:
                libero_stats = {}
                libero_stats_path = "/root/experiments/libero/libero_norm_stats.json"
                try:
                    with open(libero_stats_path, 'r') as f:
                        libero_stats = json.load(f)
                    print("Dictionary loaded from libero_norm_stats.json.")
                except FileNotFoundError:
                    print(f"Error: '{libero_stats_path}' not found. Please ensure the file exists.")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from '{libero_stats_path}'. Check file format.")
                
                self.model.norm_stats = libero_stats | self.model.norm_stats
                print("Successfully added LIBERO norm_stats to existing norm_stats of the NORA model.")

            action = self.model.inference(
                image=input_image,
                instruction=instruction,
                unnorm_key=unnorm_key,
            )
        return action

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

        # Set random seed
        set_seed_config()

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.eval_data_id]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Initialized '{self.eval_data_id}' data of for evaluation.\nNumber of tasks: {num_tasks_in_suite}")

        # Get expected image dimensions for the loaded model
        resize_dim = get_img_resize_dim(self.model_id)

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
                        replay_images.append(img)

                        # Query model to get action
                        unnorm_key = self.eval_data_id + "_no_noops"
                        action = self.model_libero_inference.remote(img, task_description, unnorm_key)
                        action = action[0]

                        # Normalize gripper action [0, 1] -> [-1, +1] as env expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        # Invert gripper action
                        if self.model_id == "nora":
                            action = invert_gripper_action(action)
                        
                        print(f"Outputted actions: {action.tolist()}")

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break

                        t += 1
                    except Exception as e:
                        raise RuntimeError(f"Caught exception: {e}")
                
                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                save_rollout_video(
                    replay_images, episode_idx, task_description, done
                )

                # Log current results
                print("-"*50)
                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                print("-"*50)
            
            # Log final results
            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

            # Save task success rate and results into dict
            if self.eval_data_id not in self.eval_summary:
                self.eval_summary[self.eval_data_id] = {}
            self.eval_summary[self.eval_data_id].update({
                task_description: {
                    "task_episodes": task_episodes,
                    "task_successes": task_successes,
                    "task_success_rate": float(task_successes) / float(task_episodes)
                }
            })
        
        # Save total success rate and results into dict
        self.eval_summary[self.eval_data_id].update({
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_success_rate": float(total_successes) / float(total_episodes)
        })

        # Save eval_summary into a JSON file
        summary_path = f'/root/vla_test/eval_summary/{self.eval_data_id}/{DATE_TIME}_{self.model_id}_finetuned={self.finetune_ok}.json'
        summary_dir = os.path.dirname(summary_path)
        os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, 'w') as json_file:
            json.dump(self.eval_summary, json_file, indent=4)

@app.local_entrypoint()
def main():
    noraSummary = ModelSummary(finetune_ok=True, num_trials_per_task=50)
    noraSummary.eval_model_on_libero.remote()