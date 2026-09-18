import pandas as pd
import os


# ======================================================
# PATH
# ======================================================

base_path = r"I:\EEE\SEM_6\FYP_EE405\Disaster"


# ======================================================
# FEATURE LIST
# ======================================================

features = [

    "blur",
    "brightness",
    "noise",
    "contrast",
    "compression",
    "sharpness",
    "exposure"

]


# ======================================================
# STORE RESULTS
# ======================================================

all_results = []


# ======================================================
# PROCESS EACH FEATURE CSV
# ======================================================

for feature in features:


    csv_path = os.path.join(
        base_path,
        f"generated_{feature}.csv"
    )


    if not os.path.exists(csv_path):

        print("Missing:", csv_path)

        continue


    print("Processing:", feature)


    df = pd.read_csv(csv_path)


    # group by level

    summary = (

        df.groupby("Level")["Feature_Value"]

        .agg(

            Count="count",

            Minimum="min",

            Maximum="max",

            Mean="mean",

            Median="median",

            Std="std"

        )

        .reset_index()

    )


    # add feature name

    summary.insert(
        0,
        "Feature",
        feature
    )


    all_results.append(summary)



# ======================================================
# COMBINE ALL FEATURES
# ======================================================


threshold_df = pd.concat(
    all_results,
    ignore_index=True
)



# ======================================================
# SAVE
# ======================================================


output_file = os.path.join(
    base_path,
    "all_feature_thresholds.csv"
)


threshold_df.to_csv(
    output_file,
    index=False
)


print("\n================================")
print("Threshold analysis completed")
print("Saved:")
print(output_file)
print("================================")

print(threshold_df)