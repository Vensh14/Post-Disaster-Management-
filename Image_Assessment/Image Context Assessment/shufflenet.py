import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"I:\EEE\FYP_EE405\Contextual_Study\Context_MobileNet_550"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
VAL_CSV   = os.path.join(BASE_PATH, "validation.csv")
TEST_CSV  = os.path.join(BASE_PATH, "test.csv")

TRAIN_DIR = os.path.join(BASE_PATH, "train")
VAL_DIR   = os.path.join(BASE_PATH, "validation")
TEST_DIR  = os.path.join(BASE_PATH, "test")

OUTPUT_DIR = os.path.join(BASE_PATH, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ============================================================
# TARGET COLUMNS
# ============================================================

TARGETS = [
    "Landslide",
    "fallen_Tree",
    "Struc_Damage",
    "Garbage",

    "Coverage",
    "Visibility",
    "Occlusion",
    "Scene Context",
    "Side",
    "Distance"
]

# ============================================================
# LOAD CSV
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("\nDataset sizes:")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASET CLASS
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, dataframe, image_folder, target, transform):

        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_name = row["Image_Name"]

        image_path = os.path.join(
            self.image_folder,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row[self.target])

        return image, label


# ============================================================
# CREATE SHUFFLENETV2
# ============================================================

def create_model(num_classes):

    # Load pretrained ShuffleNetV2 (x1.0)
    model = models.shufflenet_v2_x1_0(
        weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
    )

    # Freeze the entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classifier layer (ShuffleNetV2 uses "fc", not "classifier")
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes
    )

    # Only the new fc layer should be trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()

    # ShuffleNetV2 uses "fc", not "classifier"
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_model = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0

    for epoch in range(EPOCHS):

        # -------------------------------
        # TRAIN
        # -------------------------------

        model.train()

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = correct / total

        # -------------------------------
        # VALIDATION
        # -------------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_model)

    return model


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, test_loader, num_classes):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # ------------------------------------
    # Accuracy
    # ------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # ------------------------------------
    # Precision
    # ------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ------------------------------------
    # Recall
    # ------------------------------------

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ------------------------------------
    # F1
    # ------------------------------------

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ------------------------------------
    # AUC
    # ------------------------------------

    try:

        if num_classes == 2:

            auc = roc_auc_score(
                y_true,
                y_prob[:, 1]
            )

        else:

            # Multiclass AUC
            auc = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="weighted"
            )

    except ValueError:

        # Happens if the test set does not
        # contain all required classes
        auc = np.nan

    # ------------------------------------
    # Confusion Matrix
    # ------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    )


# ============================================================
# RUN ALL TARGETS
# ============================================================

results = []

print("\n")
print("=" * 70)
print("SHUFFLENETV2 CONTEXTUAL ANALYSIS - 550 IMAGES")
print("=" * 70)

for target in TARGETS:

    print("\n")
    print("=" * 70)
    print("TARGET:", target)
    print("=" * 70)

    # ----------------------------------------
    # Remove missing labels
    # ----------------------------------------

    target_train = train_df.dropna(
        subset=[target]
    ).copy()

    target_val = val_df.dropna(
        subset=[target]
    ).copy()

    target_test = test_df.dropna(
        subset=[target]
    ).copy()

    # ----------------------------------------
    # Convert labels to integer
    # ----------------------------------------

    target_train[target] = (
        target_train[target]
        .astype(int)
    )

    target_val[target] = (
        target_val[target]
        .astype(int)
    )

    target_test[target] = (
        target_test[target]
        .astype(int)
    )

    # ----------------------------------------
    # Determine number of classes
    # ----------------------------------------

    all_values = pd.concat([
        target_train[target],
        target_val[target],
        target_test[target]
    ])

    classes = sorted(
        all_values.unique()
    )

    num_classes = len(classes)

    print("Classes:", classes)
    print("Number of classes:", num_classes)

    if num_classes < 2:
        print(f"Only one class present for '{target}' - skipping.")
        continue

    # ----------------------------------------
    # Dataset
    # ----------------------------------------

    train_dataset = ImageDataset(
        target_train,
        TRAIN_DIR,
        target,
        train_transform
    )

    val_dataset = ImageDataset(
        target_val,
        VAL_DIR,
        target,
        test_transform
    )

    test_dataset = ImageDataset(
        target_test,
        TEST_DIR,
        target,
        test_transform
    )

    # ----------------------------------------
    # Data loaders
    # ----------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ----------------------------------------
    # Create model
    # ----------------------------------------

    model = create_model(
        num_classes
    )

    # ----------------------------------------
    # Train
    # ----------------------------------------

    model = train_model(
        model,
        train_loader,
        val_loader
    )

    # ----------------------------------------
    # Evaluate
    # ----------------------------------------

    (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    ) = evaluate_model(
        model,
        test_loader,
        num_classes
    )

    # ----------------------------------------
    # Print results
    # ----------------------------------------

    print("\nRESULTS")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if np.isnan(auc):

        print("AUC       : N/A")

    else:

        print(f"AUC       : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # ----------------------------------------
    # Save confusion matrix
    # ----------------------------------------

    safe_target = target.replace(" ", "_")

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{i}"
            for i in range(num_classes)
        ],
        columns=[
            f"Predicted_{i}"
            for i in range(num_classes)
        ]
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"ShuffleNetV2_{safe_target}_confusion_matrix_550.csv"
    )

    cm_df.to_csv(cm_path)

    # ----------------------------------------
    # Save model
    # ----------------------------------------

    model_path = os.path.join(
        OUTPUT_DIR,
        f"ShuffleNetV2_550_{safe_target}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    # ----------------------------------------
    # Add summary
    # ----------------------------------------

    results.append({
        "Feature": target,
        "Classes": num_classes,
        "Train_Images": len(target_train),
        "Validation_Images": len(target_val),
        "Test_Images": len(target_test),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "AUC": auc
    })


# ============================================================
# FINAL SUMMARY - FORMATTED FOR EASY COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Round metrics for a clean comparison table
metric_cols = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC"]
results_df[metric_cols] = results_df[metric_cols].round(4)

# Add an "Average" row across all targets for quick overall comparison
average_row = {
    "Feature": "AVERAGE",
    "Classes": "",
    "Train_Images": "",
    "Validation_Images": "",
    "Test_Images": "",
    "Accuracy": results_df["Accuracy"].mean().round(4),
    "Precision": results_df["Precision"].mean().round(4),
    "Recall": results_df["Recall"].mean().round(4),
    "F1_Score": results_df["F1_Score"].mean().round(4),
    "AUC": results_df["AUC"].mean(skipna=True).round(4)
}

comparison_df = pd.concat(
    [results_df, pd.DataFrame([average_row])],
    ignore_index=True
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "ShuffleNetV2_550_Comparison_Summary.csv"
)

comparison_df.to_csv(
    summary_path,
    index=False
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

print("\n")
print("=" * 100)
print("FINAL COMPARISON - ALL TARGETS (550 IMAGES) - SHUFFLENETV2")
print("=" * 100)

print(
    comparison_df.to_string(
        index=False
    )
)

# Ranked view: which target the model performs best/worst on by F1 score
ranked_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)[["Feature", "Accuracy", "Precision", "Recall", "F1_Score", "AUC"]]

print("\n")
print("=" * 100)
print("RANKED BY F1 SCORE (BEST TO WORST)")
print("=" * 100)

print(
    ranked_df.to_string(
        index=False
    )
)

print("\nComparison summary saved to:")
print(summary_path)

print("\nExperiment completed.")