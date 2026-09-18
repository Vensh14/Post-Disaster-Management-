# test_loading_shufflenet.py
from model_config import MODEL_CONFIG_SHUFFLENET
from model_loader import load_all_models

models = load_all_models(
    MODEL_CONFIG_SHUFFLENET,
    models_dir=r"I:\EEE\FYP_EE405\Contextual_Study\Context_MobileNet_550\results",
    device="cpu"
)
print("\nAll models loaded:", list(models.keys()))