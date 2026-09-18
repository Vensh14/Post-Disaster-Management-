import cv2
import numpy as np
import pandas as pd
import os


# =====================================================
# PATHS
# =====================================================

image_folder = r"I:\EEE\SEM_6\FYP_EE405\Disaster\Disaster_Images"

csv_path = r"C:\Users\Vensh Sambs\Desktop\Disaster data\Image_lables.csv"

output_csv = r"C:\Users\Vensh Sambs\Desktop\Disaster data\Image_lables_updated.csv"


# =====================================================
# READ CSV
# =====================================================

df = pd.read_csv(csv_path)

# remove hidden spaces in column names
df.columns = df.columns.str.strip()


print("CSV Columns:")
print(df.columns.tolist())


# =====================================================
# CREATE FEATURE COLUMNS IF NOT AVAILABLE
# =====================================================

feature_columns = [
    "Blur",
    "Brightness",
    "Noise",
    "Contrast",
    "Compression",
    "Sharpness",
    "Exposure"
]


for col in feature_columns:
    if col not in df.columns:
        df[col] = np.nan



# =====================================================
# FEATURE CALCULATION FUNCTIONS
# =====================================================

def get_blur(gray):

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()



def get_brightness(gray):

    return np.mean(gray)



def get_noise(gray):

    smooth = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    noise = (
        gray.astype(np.float32)
        -
        smooth.astype(np.float32)
    )

    return np.std(noise)



def get_contrast(gray):

    return np.std(gray)



def get_compression(img):

    _, enc = cv2.imencode(
        ".jpg",
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    size = len(enc)

    pixels = (
        img.shape[0]
        *
        img.shape[1]
    )

    return size / pixels



def get_sharpness(gray):

    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1
    )


    return np.mean(
        np.sqrt(
            gx**2 + gy**2
        )
    )



def get_exposure(gray):

    bright_pixels = np.sum(
        gray > 240
    )

    return bright_pixels / gray.size



# =====================================================
# LEVEL CONVERSION 0-4
# =====================================================

def blur_level(v):

    if v > 300:
        return 0
    elif v > 180:
        return 1
    elif v > 80:
        return 2
    elif v > 30:
        return 3
    else:
        return 4



def brightness_level(v):

    if 100 <= v <= 160:
        return 0
    elif 80 <= v < 100 or 160 < v <= 180:
        return 1
    elif 60 <= v < 80 or 180 < v <= 200:
        return 2
    elif 40 <= v < 60 or 200 < v <= 220:
        return 3
    else:
        return 4



def noise_level(v):

    if v < 5:
        return 0
    elif v < 10:
        return 1
    elif v < 20:
        return 2
    elif v < 35:
        return 3
    else:
        return 4



def contrast_level(v):

    if v > 60:
        return 0
    elif v > 45:
        return 1
    elif v > 30:
        return 2
    elif v > 20:
        return 3
    else:
        return 4



def compression_level(v):

    if v > 0.35:
        return 0
    elif v > 0.25:
        return 1
    elif v > 0.18:
        return 2
    elif v > 0.10:
        return 3
    else:
        return 4



def sharpness_level(v):

    if v > 35:
        return 0
    elif v > 25:
        return 1
    elif v > 15:
        return 2
    elif v > 8:
        return 3
    else:
        return 4



def exposure_level(v):

    if v < 0.01:
        return 0
    elif v < 0.03:
        return 1
    elif v < 0.06:
        return 2
    elif v < 0.12:
        return 3
    else:
        return 4



# =====================================================
# PROCESS IMAGES
# =====================================================

total = len(df)

print("\nProcessing", total, "images...\n")


for i in range(total):


    # Read image name
    image_name = str(
        df.at[i,"Name"]
    ).strip()


    image_path = os.path.join(
        image_folder,
        image_name
    )


    img = cv2.imread(
        image_path
    )


    if img is None:

        print(
            "Cannot read:",
            image_name
        )

        continue



    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )



    # Calculate values

    blur = get_blur(gray)

    brightness = get_brightness(gray)

    noise = get_noise(gray)

    contrast = get_contrast(gray)

    compression = get_compression(img)

    sharpness = get_sharpness(gray)

    exposure = get_exposure(gray)



    # Save levels only

    df.at[i,"Blur"] = blur_level(blur)

    df.at[i,"Brightness"] = brightness_level(brightness)

    df.at[i,"Noise"] = noise_level(noise)

    df.at[i,"Contrast"] = contrast_level(contrast)

    df.at[i,"Compression"] = compression_level(compression)

    df.at[i,"Sharpness"] = sharpness_level(sharpness)

    df.at[i,"Exposure"] = exposure_level(exposure)



    if (i+1) % 50 == 0:

        print(
            i+1,
            "/",
            total,
            "completed"
        )



# =====================================================
# CHECK RESULT
# =====================================================

print("\nSample result:")
print(df.head())



# =====================================================
# SAVE CSV
# =====================================================

df.to_csv(
    output_csv,
    index=False
)


print("\n================================")
print("Feature calculation completed")
print("CSV saved successfully")
print(output_csv)
print("================================")