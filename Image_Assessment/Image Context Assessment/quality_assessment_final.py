import os
import cv2
import numpy as np

class CombinedQualityAssessor:
    def __init__(self):
        # Empirical thresholds derived from Level 1-2 acceptable distributions
        self.thresholds = {
            'blur_min': 2.50,         
            'sharpness_min': 35.0,    
            'brightness_min': 70.0,   
            'brightness_max': 225.0,  
            'contrast_min': 25.0,     
            'noise_max': 28.0,        
            'exposure_max': 0.45      
        }

    def process_image(self, image_path):
        filename = os.path.basename(image_path)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                "filename": filename,
                "status": "REJECTED",
                "downstream_ready": False,
                "primary_issue": f"File not found at path: {image_path}",
                "all_issues": ["File path does not exist."],
                "metrics": {}
            }

        img = cv2.imread(image_path)
        if img is None:
            return {
                "filename": filename,
                "status": "REJECTED",
                "downstream_ready": False,
                "primary_issue": "File is corrupted or not a valid image.",
                "all_issues": ["Corrupt or unreadable image file."],
                "metrics": {}
            }

        # Downsample once to 512x512 for fast processing
        resized = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # PASS 1: Blur & Sharpness
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness_val = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

        # PASS 2: Brightness, Contrast, Exposure
        brightness_val = float(np.mean(gray))
        contrast_val = float(np.std(gray))
        exposure_val = float(np.sum(gray > 240) / (512.0 * 512.0))

        # PASS 3: Sensor Noise
        denoised = cv2.medianBlur(gray, 3)
        noise_val = float(np.mean(cv2.absdiff(gray, denoised)))

        # Evaluate Threshold Gates
        reasons = []

        if blur_val < self.thresholds['blur_min']:
            reasons.append(f"Blurry Image (Blur Score: {blur_val:.2f} < Min: {self.thresholds['blur_min']})")

        if sharpness_val < self.thresholds['sharpness_min']:
            reasons.append(f"Insufficient Edge Sharpness for Segmentation (Sharpness: {sharpness_val:.2f} < Min: {self.thresholds['sharpness_min']})")

        if brightness_val < self.thresholds['brightness_min']:
            reasons.append(f"Too Dark (Brightness: {brightness_val:.2f} < Min: {self.thresholds['brightness_min']})")
        elif brightness_val > self.thresholds['brightness_max']:
            reasons.append(f"Over-Exposed / Bright (Brightness: {brightness_val:.2f} > Max: {self.thresholds['brightness_max']})")

        if contrast_val < self.thresholds['contrast_min']:
            reasons.append(f"Low Contrast (Contrast: {contrast_val:.2f} < Min: {self.thresholds['contrast_min']})")

        if noise_val > self.thresholds['noise_max']:
            reasons.append(f"High Sensor Noise (Noise: {noise_val:.2f} > Max: {self.thresholds['noise_max']})")

        if exposure_val > self.thresholds['exposure_max']:
            reasons.append(f"Severe Highlight Glare (Exposure Ratio: {exposure_val:.2f} > Max: {self.thresholds['exposure_max']})")

        # Output Decision
        if len(reasons) == 0:
            status = "ACCEPTED"
            primary_issue = "Passed quality check (Level 1/2 quality). Ready for object detection & segmentation."
            downstream_ready = True
        else:
            status = "REJECTED"
            primary_issue = reasons[0]
            downstream_ready = False

        return {
            "filename": filename,
            "status": status,
            "downstream_ready": downstream_ready,
            "primary_issue": primary_issue,
            "all_issues": reasons,
            "metrics": {
                "blur": round(blur_val, 2),
                "sharpness": round(sharpness_val, 2),
                "brightness": round(brightness_val, 2),
                "contrast": round(contrast_val, 2),
                "noise": round(noise_val, 2),
                "exposure": round(exposure_val, 2)
            }
        }


# Interactive User Console
if __name__ == "__main__":
    assessor = CombinedQualityAssessor()

    print("=========================================================")
    print("      INTERACTIVE IMAGE QUALITY ASSESSMENT TOOL          ")
    print("=========================================================")
    print("Provide image paths (separated by commas for multiple files).")
    print("Example: test1.jpg, C:\\Users\\Public\\Pictures\\test2.png\n")

    user_input = input("Enter image path(s) to evaluate: ").strip()

    if not user_input:
        print("\n[!] No input provided. Exiting.")
    else:
        # Split inputs by comma to support single or multiple uploaded paths
        paths = [path.strip().strip('"').strip("'") for path in user_input.split(',')]

        print("\nProcessing images...\n")
        
        for idx, path in enumerate(paths, 1):
            result = assessor.process_image(path)
            
            print(f"---------------------------------------------------------")
            print(f" Image {idx}: {result['filename']}")
            print(f"---------------------------------------------------------")
            print(f"  STATUS:           [{result['status']}]")
            print(f"  Downstream Ready: {result['downstream_ready']}")
            print(f"  Diagnosis/Reason: {result['primary_issue']}")
            
            if len(result['all_issues']) > 1:
                print(f"  All Detected Defect(s):")
                for issue in result['all_issues']:
                    print(f"    - {issue}")
                    
            if result['metrics']:
                print(f"  Extracted Metrics:")
                for k, v in result['metrics'].items():
                    print(f"    • {k}: {v}")
            print()