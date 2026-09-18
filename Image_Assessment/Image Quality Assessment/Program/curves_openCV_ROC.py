import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv(
    r"C:\Users\Vensh Sambs\Desktop\Disaster data\Data\OpenCV_IQA_Results.csv"
)

y_true = df["Quality"]

equal_score = -df["Equal_Weight_Score"]

corr_score = -df["Correlation_Weight_Score"]

rf_score = -df["RandomForest_Weight_Score"]



fpr1, tpr1, _ = roc_curve(
    y_true,
    equal_score
)

fpr2, tpr2, _ = roc_curve(
    y_true,
    corr_score
)

fpr3, tpr3, _ = roc_curve(
    y_true,
    rf_score
)



auc1 = roc_auc_score(
    y_true,
    equal_score
)

auc2 = roc_auc_score(
    y_true,
    corr_score
)

auc3 = roc_auc_score(
    y_true,
    rf_score
)



plt.figure(figsize=(7,6))

plt.plot(
    fpr1,
    tpr1,
    label=f'Equal Weight (AUC={auc1:.3f})'
)

plt.plot(
    fpr2,
    tpr2,
    label=f'Correlation Weight (AUC={auc2:.3f})'
)

plt.plot(
    fpr3,
    tpr3,
    label=f'Random Forest Weight (AUC={auc3:.3f})'
)

plt.plot(
    [0,1],
    [0,1],
    '--',
    color='gray'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.grid(True)

plt.savefig(
    r"C:\Users\Vensh Sambs\Desktop\Disaster data\ROC_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()