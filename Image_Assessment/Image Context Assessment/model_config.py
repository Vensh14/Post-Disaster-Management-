# model_config.py

MODEL_CONFIG_MOBILENET = {
    "landslide":      {"file": "MobileNetV3_550_Landslide.pth",      "num_classes": 2, "arch": "mobilenet_v3_small"},
    "fallen_tree":    {"file": "MobileNetV3_550_fallen_Tree.pth",    "num_classes": 2, "arch": "mobilenet_v3_small"},
    "struc_damage":   {"file": "MobileNetV3_550_Struc_Damage.pth",   "num_classes": 2, "arch": "mobilenet_v3_small"},
    "garbage":        {"file": "MobileNetV3_550_Garbage.pth",        "num_classes": 2, "arch": "mobilenet_v3_small"},
    "coverage":       {"file": "MobileNetV3_550_Coverage.pth",       "num_classes": 4, "arch": "mobilenet_v3_small"},
    "visibility":     {"file": "MobileNetV3_550_Visibility.pth",     "num_classes": 4, "arch": "mobilenet_v3_small"},
    "occlusion":      {"file": "MobileNetV3_550_Occlusion.pth",      "num_classes": 4, "arch": "mobilenet_v3_small"},
    "scene_context":  {"file": "MobileNetV3_550_Scene_Context.pth",  "num_classes": 4, "arch": "mobilenet_v3_small"},
    "side":           {"file": "MobileNetV3_550_Side.pth",           "num_classes": 4, "arch": "mobilenet_v3_small"},
    "distance":       {"file": "MobileNetV3_550_Distance.pth",       "num_classes": 3, "arch": "mobilenet_v3_small"},
}

MODEL_CONFIG_SHUFFLENET = {
    "landslide":      {"file": "ShuffleNetV2_550_Landslide.pth",      "num_classes": 2, "arch": "shufflenet_v2_x1_0"},
    "fallen_tree":    {"file": "ShuffleNetV2_550_fallen_Tree.pth",    "num_classes": 2, "arch": "shufflenet_v2_x1_0"},
    "struc_damage":   {"file": "ShuffleNetV2_550_Struc_Damage.pth",   "num_classes": 2, "arch": "shufflenet_v2_x1_0"},
    "garbage":        {"file": "ShuffleNetV2_550_Garbage.pth",        "num_classes": 2, "arch": "shufflenet_v2_x1_0"},
    "coverage":       {"file": "ShuffleNetV2_550_Coverage.pth",       "num_classes": 4, "arch": "shufflenet_v2_x1_0"},
    "visibility":     {"file": "ShuffleNetV2_550_Visibility.pth",     "num_classes": 4, "arch": "shufflenet_v2_x1_0"},
    "occlusion":      {"file": "ShuffleNetV2_550_Occlusion.pth",      "num_classes": 4, "arch": "shufflenet_v2_x1_0"},
    "scene_context":  {"file": "ShuffleNetV2_550_Scene_Context.pth",  "num_classes": 4, "arch": "shufflenet_v2_x1_0"},
    "side":           {"file": "ShuffleNetV2_550_Side.pth",           "num_classes": 4, "arch": "shufflenet_v2_x1_0"},
    "distance":       {"file": "ShuffleNetV2_550_Distance.pth",       "num_classes": 3, "arch": "shufflenet_v2_x1_0"},
}