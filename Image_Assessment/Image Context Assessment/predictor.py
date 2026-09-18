# predictor.py
import torch
from torchvision import transforms
from PIL import Image

class ImagePredictor:
    def __init__(self, models_dict, device="cpu"):
        self.models = models_dict
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        results = {}

        for key, model in self.models.items():
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            results[key] = {
                "predicted_index": pred_idx,
                "confidence": round(confidence, 4)
            }

        return results