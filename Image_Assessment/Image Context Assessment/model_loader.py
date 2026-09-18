# model_loader.py
import torch
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x1_0

def build_architecture(arch_name, num_classes):
    if arch_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, num_classes
        )
    elif arch_name == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x1_0(weights=None)
        model.fc = torch.nn.Linear(
            model.fc.in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model


def load_all_models(model_config, models_dir, device="cpu"):
    loaded_models = {}

    for key, info in model_config.items():
        model = build_architecture(info["arch"], info["num_classes"])

        weight_path = f"{models_dir}/{info['file']}"
        model.load_state_dict(torch.load(weight_path, map_location=device))

        model.to(device)
        model.eval()

        loaded_models[key] = model
        print(f"Loaded: {key} ({info['arch']}, {info['num_classes']} classes)")

    return loaded_models