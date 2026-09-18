import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

csv_path = "I:/EEE/FYP_EE405/Contextual_Study/Context_labeling.csv"

# Read ONLY the first 550 image rows
df = pd.read_csv(csv_path, nrows=550)

print("Total images:", len(df))

print("\nColumns in dataset:")
print(df.columns.tolist())


# ============================================================
# 2. COLUMNS TO CHECK
# ============================================================

category_columns = [
    "Landslide",
    "fallen_Tree",
    "Struc_Damage",
    "Garbage"
]

context_columns = [
    "Coverage",
    "Visibility",
    "Occlusion",
    "Scene Context",
    "Side",
    "Distance"
]


# ============================================================
# 3. DISTRIBUTION OF DISASTER CATEGORIES
# ============================================================

print("\n" + "=" * 60)
print("DISASTER CATEGORY DISTRIBUTION")
print("=" * 60)

for col in category_columns:

    if col in df.columns:

        print(f"\n{col}")
        print("-" * 30)

        counts = df[col].value_counts(
            dropna=False
        ).sort_index()

        print("Count:")
        print(counts)

        percentages = df[col].value_counts(
            normalize=True,
            dropna=False
        ).sort_index() * 100

        print("\nPercentage:")
        print(percentages.round(2))


# ============================================================
# 4. DISTRIBUTION OF EACH CONTEXT LEVEL
# ============================================================

print("\n" + "=" * 60)
print("CONTEXT LEVEL DISTRIBUTION")
print("=" * 60)

for col in context_columns:

    if col in df.columns:

        print(f"\n{col}")
        print("-" * 30)

        counts = df[col].value_counts(
            dropna=False
        ).sort_index()

        print("Count:")
        print(counts)

        percentages = df[col].value_counts(
            normalize=True,
            dropna=False
        ).sort_index() * 100

        print("\nPercentage:")
        print(percentages.round(2))


# ============================================================
# 5. COMBINED SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary = {}

for col in category_columns + context_columns:

    if col in df.columns:

        counts = df[col].value_counts(
            dropna=False
        ).sort_index()

        summary[col] = counts


# Create table
summary_df = pd.DataFrame(summary)

# Replace only table-position NaN values with 0
# These are NOT missing dataset values.
summary_df = summary_df.fillna(0)

print(summary_df)


# ============================================================
# 6. PLOT DISTRIBUTION FOR DISASTER CATEGORIES
# ============================================================

print("\n" + "=" * 60)
print("PLOTTING DISASTER CATEGORY DISTRIBUTIONS")
print("=" * 60)

for col in category_columns:

    if col not in df.columns:
        continue

    counts = df[col].value_counts(
        dropna=False
    ).sort_index()

    plt.figure(figsize=(7, 5))

    counts.plot(kind="bar")

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. PLOT DISTRIBUTION FOR EACH CONTEXT FEATURE
# ============================================================

print("\n" + "=" * 60)
print("PLOTTING CONTEXT FEATURE DISTRIBUTIONS")
print("=" * 60)

for col in context_columns:

    if col not in df.columns:
        continue

    counts = df[col].value_counts(
        dropna=False
    ).sort_index()

    plt.figure(figsize=(7, 5))

    counts.plot(kind="bar")

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. SAVE SUMMARY TO CSV
# ============================================================

summary_df.to_csv(
    "550_images_distribution_summary.csv"
)

print("\nSummary saved as:")
print("550_images_distribution_summary.csv")