import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier


# =====================================================
# PATHS
# =====================================================

input_csv = r"C:\Users\Vensh Sambs\Desktop\Disaster data\Final_Image_Quality_Dataset.csv"

output_csv = r"C:\Users\Vensh Sambs\Desktop\Disaster data\OpenCV_IQA_Results.csv"



# =====================================================
# READ DATA
# =====================================================

df = pd.read_csv(input_csv)



features = [

    "Blur",
    "Brightness",
    "Noise",
    "Contrast",
    "Compression",
    "Sharpness",
    "Exposure"

]


X = df[features]

y = df["Quality"]



# =====================================================
# METHOD 1
# EQUAL WEIGHT SCORE
# =====================================================

df["Equal_Weight_Score"] = (

    X.sum(axis=1)

    /

    len(features)

)



# =====================================================
# METHOD 2
# CORRELATION WEIGHT
# =====================================================


correlation_weights = {

    "Sharpness":0.378516,
    "Compression":0.373992,
    "Contrast":0.366734,
    "Exposure":0.336131,
    "Blur":0.285144,
    "Brightness":0.267869,
    "Noise":0.138478

}


# normalize

total = sum(correlation_weights.values())


for key in correlation_weights:

    correlation_weights[key] /= total



df["Correlation_Weight_Score"] = 0



for feature in features:

    df["Correlation_Weight_Score"] += (

        df[feature]

        *

        correlation_weights[feature]

    )




# =====================================================
# METHOD 3
# RANDOM FOREST WEIGHT
# =====================================================


rf = RandomForestClassifier(

    n_estimators=500,

    random_state=42

)


rf.fit(X,y)


rf_weights = dict(

    zip(

        features,

        rf.feature_importances_

    )

)



df["RandomForest_Weight_Score"] = 0



for feature in features:

    df["RandomForest_Weight_Score"] += (

        df[feature]

        *

        rf_weights[feature]

    )



# =====================================================
# FIND OPTIMAL THRESHOLD
# =====================================================


def find_best_threshold(scores, labels):


    best_threshold = None

    best_f1 = 0


    thresholds = np.linspace(

        scores.min(),

        scores.max(),

        200

    )


    for t in thresholds:


        prediction = (

            scores <= t

        ).astype(int)


        f1 = f1_score(

            labels,

            prediction

        )


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = t


    return best_threshold




# =====================================================
# EVALUATION FUNCTION
# =====================================================


def evaluate_method(name, scores):


    threshold = find_best_threshold(

        scores,

        y

    )


    prediction = (

        scores <= threshold

    ).astype(int)



    accuracy = accuracy_score(

        y,

        prediction

    )


    precision = precision_score(

        y,

        prediction,

        zero_division=0

    )


    recall = recall_score(

        y,

        prediction,

        zero_division=0

    )


    f1 = f1_score(

        y,

        prediction,

        zero_division=0

    )


    auc = roc_auc_score(

        y,

        -scores

    )


    print("\n==============================")

    print(name)

    print("==============================")

    print(

        "Optimal Threshold:",

        round(threshold,4)

    )

    print(

        "Accuracy:",

        round(accuracy,4)

    )

    print(

        "Precision:",

        round(precision,4)

    )

    print(

        "Recall:",

        round(recall,4)

    )

    print(

        "F1 Score:",

        round(f1,4)

    )

    print(

        "AUC:",

        round(auc,4)

    )




# =====================================================
# PRINT RESULTS
# =====================================================


evaluate_method(

    "Equal Weight",

    df["Equal_Weight_Score"]

)


evaluate_method(

    "Correlation Weight",

    df["Correlation_Weight_Score"]

)


evaluate_method(

    "Random Forest Weight",

    df["RandomForest_Weight_Score"]

)



# =====================================================
# SAVE UPDATED CSV
# =====================================================


df.to_csv(

    output_csv,

    index=False

)


print("\nCSV saved successfully:")
print(output_csv)