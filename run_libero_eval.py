"""
Main python file for evaluating the VLA models on some data.
"""
from modal_config import gr00t_app
from policies.gr00t.gr00t_summary import GR00TSummary

@gr00t_app.local_entrypoint()
def gr00t_eval(finetune_ok: bool, eval_data_id: str, num_steps_wait: int, num_trials_per_task: int, replan_steps: int):
    gr00tSummary = GR00TSummary(
        finetune_ok = finetune_ok, 
        eval_data_id = eval_data_id, 
        num_steps_wait = num_steps_wait,
        num_trials_per_task = num_trials_per_task
    )
    gr00tSummary.eval_model_on_libero.remote()