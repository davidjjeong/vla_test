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