import os
import time
import cv2
import numpy as np

# =====================================================
# Image Folder
# =====================================================

image_folder = r"I:\EEE\SEM_6\FYP_EE405\Disaster\Disaster_Images"

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

num_images = len(image_files)

print(f"\nTotal Images : {num_images}")

# =====================================================
# Feature Functions
# (Replace with your existing functions if needed)
# =====================================================

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def noise_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    return np.std(gray.astype(np.float32)-blur.astype(np.float32))

def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def compression_score(img):
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(enc)

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F,1,0)
    gy = cv2.Sobel(gray, cv2.CV_64F,0,1)
    return np.mean(np.sqrt(gx**2+gy**2))

def exposure_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# =====================================================
# Replace these weights with YOUR actual weights
# =====================================================

corr_weights = np.array([
    0.285144,
    0.267869,
    0.138478,
    0.366734,
    0.373992,
    0.378516,
    0.336131
])

rf_weights = np.array([
    0.061122,
    0.165557,
    0.164109,
    0.132974,
    0.193354,
    0.147512,
    0.135373
])

# =====================================================
# Equal Weight Timing
# =====================================================

start = time.perf_counter()

for file in image_files:

    img = cv2.imread(os.path.join(image_folder,file))

    features = np.array([
        blur_score(img),
        brightness_score(img),
        noise_score(img),
        contrast_score(img),
        compression_score(img),
        sharpness_score(img),
        exposure_score(img)
    ])

    score = np.mean(features)

equal_total = time.perf_counter()-start

# =====================================================
# Correlation Weight Timing
# =====================================================

start = time.perf_counter()

for file in image_files:

    img = cv2.imread(os.path.join(image_folder,file))

    features = np.array([
        blur_score(img),
        brightness_score(img),
        noise_score(img),
        contrast_score(img),
        compression_score(img),
        sharpness_score(img),
        exposure_score(img)
    ])

    score = np.sum(features*corr_weights)

corr_total = time.perf_counter()-start

# =====================================================
# Random Forest Weight Timing
# =====================================================

start = time.perf_counter()

for file in image_files:

    img = cv2.imread(os.path.join(image_folder,file))

    features = np.array([
        blur_score(img),
        brightness_score(img),
        noise_score(img),
        contrast_score(img),
        compression_score(img),
        sharpness_score(img),
        exposure_score(img)
    ])

    score = np.sum(features*rf_weights)

rf_total = time.perf_counter()-start

# =====================================================
# Print Results
# =====================================================

print("\n====================================")
print("OpenCV Computational Comparison")
print("====================================")

methods = [
    ("Equal Weight", equal_total),
    ("Correlation Weight", corr_total),
    ("Random Forest Weight", rf_total)
]

for name,total in methods:

    avg = total/num_images
    throughput = num_images/total

    print(f"\n{name}")
    print("----------------------------")
    print(f"Total Time          : {total:.2f} seconds")
    print(f"Average Time/Image  : {avg:.6f} seconds")
    print(f"Average Time/Image  : {avg*1000:.2f} ms")
    print(f"Throughput          : {throughput:.2f} images/sec")
    print(f"Storage Requirement : ~0 MB")