"""
Main python file for evaluating the VLA models on some data.
"""
from modal_config import gr00t_app, nora_app
from policies.gr00t.gr00t_summary import GR00TSummary
from policies.nora.nora_summary import NoraSummary

# Execute this local entrypoint to evaluate gr00t on libero
@gr00t_app.local_entrypoint()
def gr00t_eval(finetune_ok: bool, eval_data_id: str, num_steps_wait: int, num_trials_per_task: int, replan_steps: int):
    gr00tSummary = GR00TSummary(
        finetune_ok = finetune_ok, 
        eval_data_id = eval_data_id, 
        num_steps_wait = num_steps_wait,
        num_trials_per_task = num_trials_per_task,
        replan_steps = replan_steps,
    )
    gr00tSummary.eval_model_on_libero.remote()

# Execute this local entrypoint to evaluate nora on libero
@nora_app.local_entrypoint()
def nora_eval(finetune_ok: bool, eval_data_id: str, num_steps_wait: int, num_trials_per_task: int):
    noraSummary = NoraSummary(
        finetune_ok = finetune_ok,
        eval_data_id = eval_data_id,
        num_steps_wait = num_steps_wait,
        num_trials_per_task = num_trials_per_task,
    )
    noraSummary.eval_model_on_libero.remote()