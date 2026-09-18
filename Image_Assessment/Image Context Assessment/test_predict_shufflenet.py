# test_predict_shufflenet.py
from model_config import MODEL_CONFIG_SHUFFLENET
from model_loader import load_all_models
from predictor import ImagePredictor

models = load_all_models(
    MODEL_CONFIG_SHUFFLENET,
    models_dir=r"I:\EEE\FYP_EE405\Contextual_Study\Context_MobileNet_550\results",
    device="cpu"
)

predictor = ImagePredictor(models)

result = predictor.predict(r"I:\EEE\FYP_EE405\Contextual_Study\Images\image_0650.jpg")

for key, val in result.items():
    print(f"{key}: index={val['predicted_index']}, confidence={val['confidence']}")