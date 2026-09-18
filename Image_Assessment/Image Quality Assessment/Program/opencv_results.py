import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)



# ==================================================
# PATH
# ==================================================

input_csv = r"C:\Users\Vensh Sambs\Desktop\Disaster data\Final_Image_Quality_Dataset.csv"

output_csv = r"C:\Users\Vensh Sambs\Desktop\Disaster data\OpenCV_IQA_Result.csv"



# ==================================================
# READ DATA
# ==================================================

df = pd.read_csv(input_csv)



# ==================================================
# FEATURES
# ==================================================

features = [
    "Blur",
    "Brightness",
    "Noise",
    "Contrast",
    "Compression",
    "Sharpness",
    "Exposure"
]



# ==================================================
# QUALITY SCORE
# ==================================================

# Equal importance assumption
# Each feature contributes 14.28%

weight = 1 / len(features)


df["Quality score"] = (
    df["Blur"] * weight +
    df["Brightness"] * weight +
    df["Noise"] * weight +
    df["Contrast"] * weight +
    df["Compression"] * weight +
    df["Sharpness"] * weight +
    df["Exposure"] * weight
)


df["Quality score"] = df["Quality score"].round(3)



# ==================================================
# ROC BASED THRESHOLD FINDING
# ==================================================

# Manual Quality:
# 1 = Good
# 0 = Bad
#
# But:
# Higher score means bad
#
# Therefore convert score direction

score_for_bad = df["Quality score"]



# ROC expects higher value = positive class
# Here positive = Bad

y_true_bad = 1 - df["Quality"]



auc = roc_auc_score(
    y_true_bad,
    score_for_bad
)



fpr, tpr, thresholds = roc_curve(
    y_true_bad,
    score_for_bad
)



# Best threshold using Youden index

index = np.argmax(
    tpr - fpr
)


optimal_threshold = thresholds[index]



# ==================================================
# PREDICTION
# ==================================================

df["Predicted Quality"] = np.where(
    df["Quality score"] <= optimal_threshold,
    1,
    0
)


df["Prediction Label"] = np.where(
    df["Predicted Quality"] == 1,
    "Good",
    "Bad"
)



# ==================================================
# METRICS
# ==================================================

y_true = df["Quality"]

y_pred = df["Predicted Quality"]



accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)



# ==================================================
# ADD RESULTS TO CSV
# ==================================================

df["Optimal Threshold"] = round(
    optimal_threshold,
    3
)


df["Accuracy"] = round(
    accuracy,
    4
)


df["Precision"] = round(
    precision,
    4
)


df["Recall"] = round(
    recall,
    4
)


df["F1 Score"] = round(
    f1,
    4
)


df["AUC"] = round(
    auc,
    4
)



# ==================================================
# SAVE
# ==================================================

df.to_csv(
    output_csv,
    index=False
)

print("Completed")
print("Saved:", output_csv)