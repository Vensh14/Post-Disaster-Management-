import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# ===========================================
# PATH
# ===========================================

csv_path = r"C:\Users\Vensh Sambs\Desktop\Disaster data\Final_Image_Quality_Dataset.csv"

# ===========================================
# READ DATA
# ===========================================

df = pd.read_csv(csv_path)

# ===========================================
# FEATURES
# ===========================================

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

# ===========================================
# RANDOM FOREST
# ===========================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(X, y)

# ===========================================
# FEATURE IMPORTANCE
# ===========================================

importance = rf.feature_importances_

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

# ===========================================
# NORMALIZE
# ===========================================

importance_df["Weight"] = (

    importance_df["Importance"]

    /

    importance_df["Importance"].sum()

)

print(importance_df)

# ===========================================
# SAVE
# ===========================================

importance_df.to_csv(

    r"C:\Users\Vensh Sambs\Desktop\Disaster data\RandomForest_Feature_Importance.csv",

    index=False

)

print("Done.")