import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# PATHS - CHANGE THESE IF NEEDED
# ==========================================================

csv_path = r"I:\EEE\FYP_EE405\Contextual_Study\Context_labeling.csv"

image_folder = r"I:\EEE\FYP_EE405\Contextual_Study\Images"

# A brand new output folder (won't touch or reuse any old split folder)
output_folder = r"I:\EEE\FYP_EE405\Contextual_Study\Context_MobileNet_550"

# ==========================================================
# CHECK CSV AND IMAGE FOLDER EXIST
# ==========================================================

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

if not os.path.isdir(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

print("CSV found        :", csv_path)
print("Image folder found:", image_folder)

# ==========================================================
# READ CSV
# ==========================================================

df = pd.read_csv(csv_path)

print("\nTotal CSV rows:", len(df))

# ==========================================================
# SELECT ONLY LABELED IMAGES
# ==========================================================

context_columns = [
    "Coverage",
    "Visibility",
    "Occlusion",
    "Scene Context",
    "Side",
    "Distance"
]

# Keep rows where all contextual labels exist
labeled_df = df.dropna(subset=context_columns).copy()

print("Labeled images (all context columns filled):", len(labeled_df))

# ==========================================================
# KEEP ONLY ROWS WHOSE IMAGE FILE ACTUALLY EXISTS
# ==========================================================

labeled_df["Image_Name"] = labeled_df["Image_Name"].astype(str).str.strip()

exists_mask = labeled_df["Image_Name"].apply(
    lambda name: os.path.isfile(os.path.join(image_folder, name))
)

missing_files = labeled_df.loc[~exists_mask, "Image_Name"].tolist()
if missing_files:
    print(f"\nWARNING: {len(missing_files)} labeled rows have no matching image file:")
    for name in missing_files[:20]:
        print("  -", name)
    if len(missing_files) > 20:
        print(f"  ... and {len(missing_files) - 20} more")

labeled_df = labeled_df.loc[exists_mask].reset_index(drop=True)

print("\nUsable labeled images (row + file both present):", len(labeled_df))

if len(labeled_df) < 10:
    raise ValueError("Too few usable images to split. Check csv_path / image_folder.")

# ==========================================================
# CREATE TRAIN / VALIDATION / TEST  (70% / 15% / 15%, adapts to actual count)
# ==========================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

train_df, temp_df = train_test_split(
    labeled_df,
    train_size=TRAIN_RATIO,
    random_state=42,
    shuffle=True
)

relative_val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_df, test_df = train_test_split(
    temp_df,
    train_size=relative_val_ratio,
    random_state=42,
    shuffle=True
)

print("\n====================================")
print("DATASET SPLIT")
print("====================================")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))

# ==========================================================
# CREATE NEW FOLDERS
# ==========================================================

train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "validation")
test_folder = os.path.join(output_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# ==========================================================
# FUNCTION TO COPY IMAGES
# ==========================================================

def copy_images(dataframe, destination_folder):

    copied = 0
    missing = 0

    for _, row in dataframe.iterrows():

        image_name = row["Image_Name"]

        source = os.path.join(image_folder, image_name)
        destination = os.path.join(destination_folder, image_name)

        if os.path.exists(source):

            shutil.copy2(source, destination)
            copied += 1

        else:

            print("Missing:", source)
            missing += 1

    print(
        f"{os.path.basename(destination_folder)}: "
        f"{copied} copied, {missing} missing"
    )


# ==========================================================
# COPY IMAGES
# ==========================================================

print("\nCopying images...")
copy_images(train_df, train_folder)
copy_images(val_df, val_folder)
copy_images(test_df, test_folder)

# ==========================================================
# SAVE CSV FILES
# ==========================================================

train_df.to_csv(
    os.path.join(output_folder, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(output_folder, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(output_folder, "test.csv"),
    index=False
)

print("\n====================================")
print("DATASET CREATION COMPLETED")
print("====================================")
print("Output:", output_folder)