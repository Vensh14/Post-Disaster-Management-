import pandas as pd

# Read dataset
csv_path = r"C:\Users\Vensh Sambs\Desktop\Disaster data\Final_Image_Quality_Dataset.csv"

df = pd.read_csv(csv_path)

features = [
    "Blur",
    "Brightness",
    "Noise",
    "Contrast",
    "Compression",
    "Sharpness",
    "Exposure"
]

# Pearson correlation with Quality
correlations = []

for feature in features:

    r = df[feature].corr(df["Quality"])

    correlations.append({
        "Feature": feature,
        "Correlation": r,
        "Absolute Correlation": abs(r)
    })

corr_df = pd.DataFrame(correlations)

corr_df = corr_df.sort_values(
    "Absolute Correlation",
    ascending=False
)

print(corr_df)

corr_df.to_csv(
    r"C:\Users\Vensh Sambs\Desktop\Disaster data\Feature_Correlation.csv",
    index=False
)