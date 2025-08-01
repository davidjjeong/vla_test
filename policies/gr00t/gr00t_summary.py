import modal
import os
import json
import collections

from modal_config import data_vol, rollouts_vol, eval_summary_vol, gr00t_image, gr00t_app
from experiments.eval_utils import (
    set_seed_config,
    save_rollout_video,
    normalize_gripper_action,
    invert_gripper_action,
    quat2axisangle,
    DATE,
    DATE_TIME
)
from experiments.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_img_resize_dim,
    get_preprocessed_img
)
from .gr00t_utils import unchunk

# Import necessary packages
with gr00t_image.imports():
    import torch
    import tqdm
    import numpy as np
    from PIL import Image

    from vla_test.data.libero.libero import benchmark
    from vla_test.models.gr00t.gr00t.model.policy import Gr00tPolicy
    from vla_test.models.gr00t.gr00t.experiment.data_config import DATA_CONFIG_MAP

@gr00t_app.cls(
    image=gr00t_image,
    volumes={
        "/root/vla_test/data/libero/datasets": data_vol,
        "/root/vla_test/rollouts": rollouts_vol,
        "/root/vla_test/eval_summary": eval_summary_vol,
    },
    gpu="A100",
    timeout=60*60*24,   # 24hr timeout
    retries=3
)
class GR00TSummary():
    """
    Modal class to load and evaluate various pre-trained checkpoints of GR00T.
    Instance of this class will persist in the cloud, allowing the model to
    be loaded once and be reused again.
    """

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # LIBERO-specific GR00T configuration
        if self.eval_data_id.split("_")[0] == "libero":
            embodiment_tag = "new_embodiment"
            data_config = DATA_CONFIG_MAP["libero"]
            modality_config = data_config.modality_config()
            modality_transform = data_config.transform()

        # --- Load Model ---
        if self.finetune_ok:
            repo_id = "delinqu/gr00t-libero-goal"
        else:
            repo_id = "nvidia/GR00T-N1.5-3B"

        try:
            print(f"Loading VLA model '{repo_id}'...")
            self.model = Gr00tPolicy(
                model_path=repo_id,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=device
            )
            print(f"Successfully loaded '{repo_id}'!")
        except Exception as e:
            raise RuntimeError(f"Failed to load '{repo_id}': {e}")
    
    @modal.method()
    def model_libero_inference(self, observation):
        """
        Query model to generate robot action. Return the actions outputted by the model.
        """
        action = self.model.get_action(observation)
        return action
    
    @modal.method()
    def eval_model_on_libero(self):
        """
        Evaluate specified LIBERO task suite on the loaded GR00T model.
        Evaluation code inspired by and modified from [OpenVLA]: https://github.com/openvla/openvla
        """
        if self.model is None:
            raise RuntimeError("Cannot load model for evaluation. Check if repo_id exists in Hugging Face.")

        # Set random seed
        set_seed_config()

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.eval_data_id]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Initialized '{self.eval_data_id}' data of for evaluation.\nNumber of tasks: {num_tasks_in_suite}")

        # Get expected image dimensions for the loaded model
        resize_dim = get_img_resize_dim("gr00t")

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
                action_plan = collections.deque()

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

                        # Rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Compute new action chunk if the robot executed all actions of the previous chunk
                            observation = {
                                "video.image": np.expand_dims(img, axis=0),
                                "video.wrist_image": np.expand_dims(wrist_img, axis=0),
                                "state.state": np.expand_dims(
                                    np.concatenate(
                                        obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
                                    ),
                                    axis=0
                                ),
                                "annotation.human.task_description": [task_description],
                            }

                            # Query model to get action
                            action_chunk = self.model_libero_inference.remote(observation)
                            action_plan = unchunk(action_chunk, action_plan, replan_steps=16)
                        
                        action = action_plan.popleft()
                        # Normalize gripper action [0, 1] -> [-1, +1] as env expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        # Invert gripper action
                        action = invert_gripper_action(action)
                        
                        #print(f"Outputted actions: {action.tolist()}")

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
        summary_path = f'/root/vla_test/eval_summary/{self.eval_data_id}/{DATE_TIME}_gr00t_finetuned={self.finetune_ok}.json'
        summary_dir = os.path.dirname(summary_path)
        os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, 'w') as json_file:
            json.dump(self.eval_summary, json_file, indent=4)

@gr00t_app.local_entrypoint()
def main():
    gr00tSummary = GR00TSummary(finetune_ok=True, num_trials_per_task=50)
    gr00tSummary.eval_model_on_libero.remote()