"""
Main python file for evaluating the VLA models on some data.
"""
from modal_config import gr00t_app
from policies.gr00t.gr00t_summary import GR00TSummary

@gr00t_app.local_entrypoint()
def gr00t_eval():
    gr00tSummary = GR00TSummary(finetune_ok=True)
    gr00tSummary.eval_model_on_libero.remote()