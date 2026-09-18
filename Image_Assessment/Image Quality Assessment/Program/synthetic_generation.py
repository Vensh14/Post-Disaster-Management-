import cv2
import os
import numpy as np
import random
from tqdm import tqdm


# ==================================================
# PATHS
# ==================================================

input_folder = r"I:/EEE/SEM_6/FYP_EE405/Disaster/Disaster_Images"

output_folder = r"I:/EEE/SEM_6/FYP_EE405/Disaster/generated_brightness"

os.makedirs(output_folder, exist_ok=True)



# ==================================================
# DISTORTION FUNCTIONS
# ==================================================

def add_blur(img, level):

    kernels = {
        1:7,
        2:15,
        3:31,
        4:51
    }

    k = kernels[level]

    return cv2.GaussianBlur(
        img,
        (k,k),
        0
    )



def add_brightness(img, level):

    value = {
        1:30,
        2:70,
        3:-70,
        4:-140
    }

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    hsv[:,:,2] = np.clip(
        hsv[:,:,2].astype(np.int16) + value[level],
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )



def add_noise(img, level):

    amount = {
        1:10,
        2:30,
        3:70,
        4:100
    }

    noise = np.random.normal(
        0,
        amount[level],
        img.shape
    )

    noisy = img + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)



def add_contrast(img, level):

    alpha = {
        1:0.8,
        2:0.6,
        3:0.4,
        4:0.2
    }

    return cv2.convertScaleAbs(
        img,
        alpha=alpha[level],
        beta=128*(1-alpha[level])
    )



def add_compression(img, level):

    quality = {
        1:40,
        2:20,
        3:8,
        4:2
    }

    params = [
        cv2.IMWRITE_JPEG_QUALITY,
        quality[level]
    ]

    _, encoded = cv2.imencode(
        ".jpg",
        img,
        params
    )

    return cv2.imdecode(
        encoded,
        1
    )



def add_sharpness(img, level):

    blur = cv2.GaussianBlur(
        img,
        (5,5),
        level*2
    )

    amount = {
        1:0.3,
        2:0.5,
        3:0.7,
        4:0.9
    }

    return cv2.addWeighted(
        img,
        1-amount[level],
        blur,
        amount[level],
        0
    )



def add_exposure(img, level):

    value = {
        1:40,
        2:80,
        3:120,
        4:160
    }

    return np.clip(
        img.astype(np.int16)+value[level],
        0,
        255
    ).astype(np.uint8)



# ==================================================
# FEATURE DICTIONARY
# ==================================================

functions = {
    
    "brightness": add_brightness,

}



# ==================================================
# READ ORIGINAL IMAGES
# ==================================================

images = [
    x for x in os.listdir(input_folder)
    if x.lower().endswith(
        (".jpg",".jpeg",".png")
    )
]


print("Original images:", len(images))


random.shuffle(images)



# ==================================================
# SAVE FUNCTION
# ==================================================

def save_image(img, filename):

    cv2.imwrite(
        os.path.join(
            output_folder,
            filename
        ),
        img
    )



# ==================================================
# PART 1
# SINGLE FEATURE GENERATION
# ==================================================

total = 0


print("\nGenerating individual features...")


for feature, func in functions.items():

    print("\nFeature:", feature)


    for level in range(1,5):

        print(
            "Level:",
            level
        )


        selected = random.sample(
            images,
            100
        )


        for image_name in tqdm(selected):

            img_path = os.path.join(
                input_folder,
                image_name
            )


            img = cv2.imread(
                img_path
            )


            if img is None:
                continue



            output = func(
                img.copy(),
                level
            )


            original_name = os.path.splitext(
                image_name
            )[0]


            filename = (
                f"{original_name}_"
                f"{feature}_"
                f"L{level}.jpg"
            )


            save_image(
                output,
                filename
            )


            total += 1



'''
# ==================================================
# PART 2
# MIXED DISTORTION GENERATION
# ==================================================

print("\nGenerating mixed distortions...")


for level in range(1,5):

    print(
        "Mixed Level:",
        level
    )


    selected = random.sample(
        images,
        250
    )


    for image_name in tqdm(selected):


        img = cv2.imread(
            os.path.join(
                input_folder,
                image_name
            )
        )


        if img is None:
            continue



        mixed = img.copy()



        for feature, func in functions.items():

            mixed = func(
                mixed,
                level
            )



        original_name = os.path.splitext(
            image_name
        )[0]


        filename = (
            f"{original_name}_"
            f"MIXED_"
            f"L{level}.jpg"
        )


        save_image(
            mixed,
            filename
        )


        total += 1
'''


# ==================================================
# RESULT
# ==================================================

print("\n============================")
print("Generation Completed")
print("============================")

print(
    "Generated images:",
    total
)