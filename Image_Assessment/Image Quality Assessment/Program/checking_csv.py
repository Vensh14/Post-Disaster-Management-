import pandas as pd
import os


# =====================================================
# PATH
# =====================================================

csv_path = r"C:/Users/Vensh Sambs/Desktop/Disaster data/Final_Image_Quality_Dataset.csv"

image_folder = r"I:/EEE/SEM_6/FYP_EE405/Disaster"


# =====================================================
# READ CSV
# =====================================================

df = pd.read_csv(csv_path)


print("==============================")
print("DATASET INFORMATION")
print("==============================")

print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# =====================================================
# 1. CHECK DUPLICATE IMAGE NAMES
# =====================================================

print("\n==============================")
print("DUPLICATE IMAGE NAMES")
print("==============================")


duplicates = df[
    df["Name"].duplicated()
]


if len(duplicates) == 0:

    print("No duplicate image names")

else:

    print(
        "Duplicate count:",
        len(duplicates)
    )

    print(
        duplicates[["Image_Num","Name"]]
    )



# =====================================================
# 2. CHECK DUPLICATE IMAGE NUMBERS
# =====================================================

print("\n==============================")
print("DUPLICATE IMAGE NUMBERS")
print("==============================")


dup_numbers = df[
    df["Image_Num"].duplicated()
]


if len(dup_numbers)==0:

    print("No duplicate Image_Num")

else:

    print(dup_numbers)



# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\n==============================")
print("MISSING VALUES")
print("==============================")


missing = df.isnull().sum()


print(missing)


# =====================================================
# 4. CHECK FEATURE LEVEL RANGE
# =====================================================

print("\n==============================")
print("FEATURE LEVEL CHECK")
print("==============================")


features = [
    "Blur",
    "Brightness",
    "Noise",
    "Contrast",
    "Compression",
    "Sharpness",
    "Exposure",
    "MIXED"
]


for feature in features:

    wrong = df[
        ~df[feature].isin([1,2,3,4])
    ]


    print(
        feature,
        "wrong values:",
        len(wrong)
    )



# =====================================================
# 5. QUALITY DISTRIBUTION
# =====================================================

print("\n==============================")
print("QUALITY DISTRIBUTION")
print("==============================")


print(
    df["Quality"].value_counts()
)



# =====================================================
# 6. VISIBILITY DISTRIBUTION
# =====================================================

print("\n==============================")
print("VISIBILITY DISTRIBUTION")
print("==============================")


print(
    df["Visibility"].value_counts()
)



# =====================================================
# 7. CHECK IMAGE FILE EXISTENCE
# =====================================================

print("\n==============================")
print("IMAGE FILE CHECK")
print("==============================")


missing_images=[]


for name in df["Name"]:

    found=False


    for folder in [
        "Disaster_Images",
        "Generated_Images",
        "generated_blur",
        "generated_brightness",
        "generated_noise",
        "generated_contrast",
        "generated_compression",
        "generated_sharpness",
        "generated_exposure"
    ]:

        path=os.path.join(
            image_folder,
            folder,
            name
        )


        if os.path.exists(path):

            found=True
            break


    if not found:

        missing_images.append(name)



print(
    "Missing image files:",
    len(missing_images)
)


if len(missing_images)>0:

    print(
        missing_images[:10]
    )



# =====================================================
# 8. IMAGE NUMBER ORDER CHECK
# =====================================================

print("\n==============================")
print("IMAGE NUMBER CHECK")
print("==============================")


numbers=df["Image_Num"].values


expected=range(
    1,
    len(df)+1
)


if list(numbers)==list(expected):

    print(
        "Image numbers are continuous"
    )

else:

    print(
        "Image numbers have gaps"
    )



print("\n==============================")
print("CHECK COMPLETED")
print("==============================")