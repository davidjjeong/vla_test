import modal
import os

# Define base Modal image
vla_image = (
    modal.Image.debian_slim()
    .pip_install(
        "datasets",
        "transformers",
        "torch",
        "Pillow",
    )
    .env({
        "HF_HOME": "/root/data",
    })
)

with vla_image.imports():
    from datasets import load_dataset
    import transformers

# Define Modal volume
vol = modal.Volume.from_name("eval_data", create_if_missing=True)

# Define Modal app
app = modal.App(
    name="vla-model-report",
    image = vla_image,
    secrets=[modal.Secret.from_name("vla-eval-secret")]
)

@app.cls(
    image=vla_image,
    volumes={"/root/data": vol},
)
class ModelReport():
    """
    Modal class to load and evaluate Vision-Language-Action (VLA) models.
    Instance of this class will persist in the cloud, allowing the model to
    be loaded once and be reused again.
    """
    model_id: str = modal.parameter(default="nora")
    eval_data_id: str = modal.parameter(default="libero_spatial_image")

    @modal.enter()
    def init_report(self):
        # Initialize variables
        self.model_id = self.model_id.lower()                   # label of the VLA model
        self.model = None                                       # stores the loaded model object
        self.eval_summary = {}                                  # stores the evaluation summary
        self.eval_data_dir = "lerobot/" + self.eval_data_id     # path to specified dataset on Hugging Face(HF)
        self.hf_token = os.environ.get("HF_TOKEN")              # token to access models and datasets on HF

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
            print(f"'{self.eval_data_id}' successfully loaded!")
        except Exception as e:
            raise RuntimeError(f"Error loading '{self.eval_data_id}' from Hugging Face: {e}")

@app.function(image=vla_image)
def main():
    noraReport = ModelReport()
    noraReport.load_eval_data.remote()