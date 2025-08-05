
# **VLA Preliminary Model Evaluation**

Given the lack of previous open-sourced work to evaluate vision-language-action (VLA) models on a common set of data with identical
set-up, this project aims to evaluate three open-sourced state-of-the-art (SOTA) VLA models - NORA, Pi-0, GR00T N1.5 - shortlisted from a comparison of 15 VLA models conducted by literature review with a primary focus on **low computational overhead** and **open-source availability**.

The repo contains reproducible code to evaluate NORA, Pi-0, and GR00T N1.5 (currently NORA only) on simulation data, specifically the LIBERO benchmark which has four task suites (Spatial, Object, Goal, Long).

Considering the low success rate of the pre-trained models without fine-tuning, we run only 10 episodes for each task to reduce computation. We run 50 episodes per task for finetuned pre-trained models.

## Getting Started

We use [Modal](https://modal.com/) to run our functions remotely on a GPU-supported cloud. To get started, you need to set-up a Modal account. Please follow the instructions [here](https://modal.com/signup).

Once the account is set-up, clone the repo:
```bash
git clone https://github.com/davidjjeong/vla_test.git
cd vla_test
```

Create a new conda environment and install Modal. No need to install other dependencies as they will be installed when building the Modal image.
```bash
# Create and activate conda environment
conda create -n modal_env python=3.10 -y
conda activate modal_env
pip install modal
python -m modal setup
```

## 1. Download LIBERO Data

Before we start evaluating the VLA models, we need to download the LIBERO dataset into the cloud environment.

Since we use Hugging Face to download the data, you need to set up an user access token on Hugging Face. You can refer to the instructions [here](https://huggingface.co/docs/hub/en/security-tokens). The access token can be read-only. Copy the value of the access token and save it in a safe place where you can remember.

Now, we need to create a secret in your Modal workspace to register your access token. Navigate to **Secrets** tab in your workspace and click the button **Create new secret** as shown below.

<div align="center">
<img src="assets/secret_setup_step_1.png" width="800" alt="Setup Modal Secret - Step 1">
</div>

Then, select **Custom** type for your secret.

<div align="center">
<img src="assets/secret_setup_step_2.png" width="800" alt="Setup Modal Secret - Step 2">
</div>

Follow the configurations as shown in the image below, and paste your copied access token in the **Value** entry.

<div align="center">
<img src="assets/secret_setup_step_3.png" width="800" alt="Setup Modal Secret - Step 3">
</div>

Now you can simply run this command below to download the data. You only need to execute this once, as the data will be stored permanently in the Modal volume of your workspace.
```bash
modal run experiments/libero/libero_utils.py::download_libero_nora
```

## 2. Evaluate VLA Model

Once you have downloaded LIBERO data, you are all set to evaluate your desired VLA model.

Currently, the default setup uses a A100 40GB GPU. If you would like to change this configuration, you can specify a `gpu` in the Modal app class definition of respective VLA model evaluation classes.

For NORA, refer to [`nora_summary.py`](./policies/nora/nora_summary.py). For GR00T, refer to [`gr00t_summary.py`](./policies/gr00t/gr00t_summary.py).

For instance, this is how the Modal app definition of `GR00TSummary` class is defined in [`gr00t_summary.py`](./policies/gr00t/gr00t_summary.py):
```python
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
    ...
```
You can specify `gpu` from a variety of GPU options that Modal support: `["T4", "L4", "A10", "A100", "A100-40GB", "A100-80GB", "L40S", "H100 / H100!", "H200", "B200"]`. You can refer to the [Modal docs](https://modal.com/docs/guide/gpu#specifying-gpu-type) for greater detail in specifying gpu type and count.


To evaluate a fine-tuned NORA pre-trained model on LIBERO Object with 50 rollouts per task, simply run this command below in your terminal:
```bash
modal run run_libero_eval.py::nora_eval --finetune-ok \
                                        --eval-data-id "libero_object" \
                                        --num-steps-wait 10 \
                                        --num-trials-per-task 50
```

To evaluate GR00T on LIBERO, there is another parameter `replan_steps`, which represents the action chunk size.
By default, GR00T uses `replan_steps = 16`.

The following terminal command will run a fine-tuned gr00t model on LIBERO Goal with no action chunking and 10 rollouts per task:
```bash
modal run run_libero_eval.py::gr00t_eval --finetune-ok \
                                         --eval-data-id "libero_goal" \
                                         --num-steps-wait 10 \
                                         --num-trials-per-task 10 \
                                         --replan-steps 1
```