import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Import MobileNet context prediction dependencies
from model_config import MODEL_CONFIG_MOBILENET
from model_loader import load_all_models
from predictor import ImagePredictor


# =============================================================================
# HUMAN-READABLE DICTIONARIES FOR YOUR 10 SPECIFIC MODELS (EDIT AS NEEDED)
# =============================================================================

# 2-Class Models (Binary Detection)
LANDSLIDE_LABELS = {
    0: "No Landslide Detected",
    1: "Landslide Detected"
}

FALLEN_TREE_LABELS = {
    0: "No Fallen Trees",
    1: "Fallen Tree / Debris Detected"
}

STRUC_DAMAGE_LABELS = {
    0: "No Structural Damage",
    1: "Structural Damage Detected"
}

GARBAGE_LABELS = {
    0: "No Garbage / Waste Clear",
    1: "Garbage / Waste Accumulation"
}

# 4-Class Models
COVERAGE_LABELS = {
    0: "Very low Coverage",
    1: "Low Coverage",
    2: "Moderate Coverage",
    3: "HighCoverage"
}

VISIBILITY_LABELS = {
    0: "Cannot identify affected area",
    1: "Hard to identify",
    2: "Mostly visible",
    3: "Clear Visibility"
}

OCCLUSION_LABELS = {
    0: "Fully hid or cannot identify",
    1: "mostly hid",
    2: "some of the affected area may be hid",
    3: "No hidden parts"
}

SCENE_CONTEXT_LABELS = {
    0: "No useful surrounding information",
    1: "Limited surrounding information",
    2: "Sufficient surrounding information",
    3: "Clear surrounding information"
}

SIDE_LABELS = {
    0: "Left View",
    1: "Straight View",
    2: "Right View",
    3: "Other View"
}

# 3-Class Model
DISTANCE_LABELS = {
    0: "Very near",
    1: "Allowable range",
    2: "Far"
}


# User-Friendly Explanations for Common Rejection Reasons
USER_FRIENDLY_ISSUES = {
    "blur": "Image is too blurry or out of focus.",
    "sharpness_low": "Details are too soft to recognize features.",
    "sharpness_high": "Image contains high noise or unnatural digital artifacts.",
    "dark": "Lighting is too dark to clearly identify objects.",
    "bright": "Image is washed out or overexposed.",
    "contrast": "Image has poor contrast (washed out tones).",
    "glare": "Excessive glare or strong light reflections present.",
    "noise": "Image has heavy digital grain/sensor noise.",
    "file_error": "File missing, unreadable, or corrupted format."
}


# =============================================================================
# 1. OPENCV QUALITY ASSESSMENT PIPELINE
# =============================================================================
class QualityAssessor:
    def __init__(self):
        # Balanced empirical thresholds for real-world & disaster imagery
        self.thresholds = {
            'blur_min': 2.50,           # Filters unusable motion blur
            'sharpness_min': 25.0,      # Lower bound for edge boundaries
            'clarity_max': 850.0,       # Upper cap to avoid false rejection of sharp rubble/trees
            'brightness_min': 40.0,     # Accepts low-light/shadow disaster scenes
            'brightness_max': 230.0,    # Rejects severely washed-out whiteouts
            'contrast_min': 18.0,       # Ensures sufficient dynamic range
            'noise_max': 32.0,          # Accommodates camera sensor noise
            'exposure_max': 0.45        # Max allowed ratio of clipped white pixels (>240)
        }

    def process(self, image_path):
        if not os.path.exists(image_path):
            return {
                "status": "REJECTED",
                "downstream_ready": False,
                "user_summary": USER_FRIENDLY_ISSUES["file_error"],
                "technical_reasons": ["File path does not exist."],
                "metrics": {}
            }

        img = cv2.imread(image_path)
        if img is None:
            return {
                "status": "REJECTED",
                "downstream_ready": False,
                "user_summary": USER_FRIENDLY_ISSUES["file_error"],
                "technical_reasons": ["Corrupt or unreadable image file."],
                "metrics": {}
            }

        # Standardize resolution for uniform evaluation
        resized = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Metric Calculations
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness_val = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

        brightness_val = float(np.mean(gray))
        contrast_val = float(np.std(gray))
        exposure_val = float(np.sum(gray > 240) / (512.0 * 512.0))

        denoised = cv2.medianBlur(gray, 3)
        noise_val = float(np.mean(cv2.absdiff(gray, denoised)))

        # Evaluate Quality Gates
        tech_reasons = []
        user_reasons = []

        if blur_val < self.thresholds['blur_min']:
            tech_reasons.append(f"Excessive Blur (Blur: {blur_val:.2f} < Min: {self.thresholds['blur_min']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["blur"])

        if sharpness_val < self.thresholds['sharpness_min']:
            tech_reasons.append(f"Low Edge Sharpness (Sharpness: {sharpness_val:.2f} < Min: {self.thresholds['sharpness_min']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["sharpness_low"])
        elif sharpness_val > self.thresholds['clarity_max']:
            tech_reasons.append(f"Unnatural Clutter / Edge Noise (Sharpness: {sharpness_val:.2f} > Max: {self.thresholds['clarity_max']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["sharpness_high"])

        if brightness_val < self.thresholds['brightness_min']:
            tech_reasons.append(f"Too Dark (Brightness: {brightness_val:.2f} < Min: {self.thresholds['brightness_min']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["dark"])
        elif brightness_val > self.thresholds['brightness_max']:
            tech_reasons.append(f"Over-Exposed (Brightness: {brightness_val:.2f} > Max: {self.thresholds['brightness_max']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["bright"])

        if contrast_val < self.thresholds['contrast_min']:
            tech_reasons.append(f"Low Contrast (Contrast: {contrast_val:.2f} < Min: {self.thresholds['contrast_min']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["contrast"])

        if exposure_val > self.thresholds['exposure_max']:
            tech_reasons.append(f"Severe Highlight Glare (Glare Ratio: {exposure_val:.2f} > Max: {self.thresholds['exposure_max']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["glare"])

        if noise_val > self.thresholds['noise_max']:
            tech_reasons.append(f"High Sensor Noise (Noise: {noise_val:.2f} > Max: {self.thresholds['noise_max']})")
            user_reasons.append(USER_FRIENDLY_ISSUES["noise"])

        if len(tech_reasons) == 0:
            status = "ACCEPTED"
            user_summary = "Image quality is acceptable"
            downstream_ready = True
        else:
            status = "REJECTED"
            user_summary = user_reasons[0]
            downstream_ready = False

        return {
            "status": status,
            "downstream_ready": downstream_ready,
            "user_summary": user_summary,
            "user_reasons": user_reasons,
            "technical_reasons": tech_reasons,
            "metrics": {
                "blur": round(blur_val, 2),
                "sharpness": round(sharpness_val, 2),
                "brightness": round(brightness_val, 2),
                "contrast": round(contrast_val, 2),
                "noise": round(noise_val, 2),
                "exposure": round(exposure_val, 2)
            }
        }


# =============================================================================
# 2. INTERACTIVE MULTI-PHOTO SELECTION WORKFLOW
# =============================================================================
def collect_user_images():
    temp_root = tk.Tk()
    temp_root.withdraw()
    temp_root.attributes('-topmost', True)

    image_paths = []
    while True:
        path = filedialog.askopenfilename(
            title=f"Select Photo #{len(image_paths) + 1}",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )

        if path:
            image_paths.append(path)
            add_more = messagebox.askyesno(
                "Add Another Photo?",
                f"Selected: {os.path.basename(path)}\n\nDo you want to add another photo?"
            )
            if not add_more:
                break
        else:
            break

    temp_root.destroy()
    return image_paths


# =============================================================================
# 3. HELPER FUNCTION TO PARSE PREDICTIONS TO CLEAN USER TEXT
# =============================================================================
def format_user_findings(context_data):
    lines = []

    # Map each of your 10 output categories specifically
    mappings = {
        'landslide': LANDSLIDE_LABELS,
        'fallen_tree': FALLEN_TREE_LABELS,
        'struc_damage': STRUC_DAMAGE_LABELS,
        'garbage': GARBAGE_LABELS,
        'coverage': COVERAGE_LABELS,
        'visibility': VISIBILITY_LABELS,
        'occlusion': OCCLUSION_LABELS,
        'scene_context': SCENE_CONTEXT_LABELS,
        'side': SIDE_LABELS,
        'distance': DISTANCE_LABELS
    }

    for key, label_dict in mappings.items():
        if key in context_data:
            idx = context_data[key]['predicted_index']
            clean_title = key.replace('_', ' ').title()
            text_val = label_dict.get(idx, f"Class {idx}")
            lines.append(f"• {clean_title:<14}: {text_val}")

    return "\n".join(lines) if lines else "Assessment completed."


# =============================================================================
# 4. USER-FRIENDLY SUBPLOT DASHBOARD (INTEGER FONTS FIXED)
# =============================================================================
class RowSubplotDashboard:
    def __init__(self, results):
        self.root = tk.Tk()
        self.root.title("Disaster Assessment Dashboard")
        self.root.geometry("1150x700")

        # Scrollable container setup
        canvas = tk.Canvas(self.root, bg="#F8FAFC")
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title Header
        header = tk.Label(
            scrollable_frame, 
            text="Context Assessment Summary", 
            font=("Arial", 16, "bold"), 
            fg="#0F172A", 
            bg="#F8FAFC"
        )
        header.grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(15, 10))

        # Column Headers
        th_photo = tk.Label(
            scrollable_frame, 
            text="PHOTO", 
            font=("Arial", 11, "bold"), 
            bg="#E2E8F0", 
            fg="#1E293B", 
            width=20, 
            pady=6
        )
        th_photo.grid(row=1, column=0, padx=(20, 5), pady=5, sticky="ew")

        th_quality = tk.Label(
            scrollable_frame, 
            text="QUALITY CHECK", 
            font=("Arial", 11, "bold"), 
            bg="#E2E8F0", 
            fg="#1E293B", 
            width=30, 
            pady=6
        )
        th_quality.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        th_context = tk.Label(
            scrollable_frame, 
            text="CONTEXT FINDINGS", 
            font=("Arial", 11, "bold"), 
            bg="#E2E8F0", 
            fg="#1E293B", 
            width=55, 
            pady=6
        )
        th_context.grid(row=1, column=2, padx=(5, 20), pady=5, sticky="ew")

        # Render Rows Dynamically
        for idx, item in enumerate(results, start=2):
            # --- COLUMN 1: PHOTO ---
            cell_photo = tk.LabelFrame(scrollable_frame, bg="#FFFFFF", bd=1, relief="solid")
            cell_photo.grid(row=idx, column=0, padx=(20, 5), pady=8, sticky="nsew")

            fn_lbl = tk.Label(
                cell_photo, 
                text=item['filename'], 
                font=("Arial", 9, "bold"), 
                bg="#FFFFFF", 
                fg="#0F172A", 
                wraplength=140
            )
            fn_lbl.pack(padx=5, pady=(5, 2))

            try:
                pil_img = Image.open(item['path'])
                pil_img.thumbnail((140, 140))
                tk_img = ImageTk.PhotoImage(pil_img)
                img_lbl = tk.Label(cell_photo, image=tk_img, bg="#FFFFFF")
                img_lbl.image = tk_img
                img_lbl.pack(padx=5, pady=5)
            except Exception:
                err_lbl = tk.Label(cell_photo, text="[Image Load Error]", bg="#FEE2E2", fg="#991B1B", width=16, height=7)
                err_lbl.pack(padx=5, pady=5)

            # --- COLUMN 2: QUALITY CHECK ---
            q = item['quality']
            status_color = "#15803D" if q['status'] == "ACCEPTED" else "#B91C1C"
            cell_quality = tk.LabelFrame(scrollable_frame, bg="#FFFFFF", bd=1, relief="solid")
            cell_quality.grid(row=idx, column=1, padx=5, pady=8, sticky="nsew")

            q_status = tk.Label(
                cell_quality, 
                text=f"STATUS: {q['status']}", 
                font=("Arial", 11, "bold"), 
                fg=status_color, 
                bg="#FFFFFF"
            )
            q_status.pack(anchor="w", padx=10, pady=(12, 4))

            q_diag = tk.Label(
                cell_quality, 
                text=q['user_summary'], 
                font=("Arial", 9), 
                fg="#334155", 
                bg="#FFFFFF", 
                wraplength=210, 
                justify="left"
            )
            q_diag.pack(anchor="w", padx=10, pady=(0, 10))

            # --- COLUMN 3: CONTEXT FINDINGS ---
            cell_context = tk.LabelFrame(scrollable_frame, bg="#FFFFFF", bd=1, relief="solid")
            cell_context.grid(row=idx, column=2, padx=(5, 20), pady=8, sticky="nsew")

            if item['context'] is not None:
                parsed_text = format_user_findings(item['context'])
                c_lbl = tk.Label(
                    cell_context, 
                    text=parsed_text, 
                    font=("Consolas", 9), 
                    fg="#0F172A", 
                    bg="#FFFFFF", 
                    justify="left",
                    anchor="w"
                )
                c_lbl.pack(anchor="w", padx=12, pady=10)
            else:
                skip_lbl = tk.Label(
                    cell_context, 
                    text="Assessment Skipped\n\nThis image was rejected by quality gate.", 
                    font=("Arial", 9, "italic"), 
                    fg="#991B1B", 
                    bg="#FEF2F2", 
                    justify="center", 
                    pady=20
                )
                skip_lbl.pack(fill="both", expand=True, padx=10, pady=10)

    def show(self):
        self.root.mainloop()


# =============================================================================
# 5. TERMINAL PRINTING (TECHNICAL METRICS & CONFIDENCE VALUES)
# =============================================================================
def print_terminal_logs(item):
    print("\n" + "="*80)
    print(f"FILE: {item['filename']}")
    print(f"PATH: {item['path']}")
    print("-" * 80)
    
    q = item['quality']
    print(f"[QUALITY ASSESSMENT]")
    print(f"  Status       : {q['status']}")
    print(f"  User Summary : {q['user_summary']}")
    
    if q['metrics']:
        print("  Metrics      :")
        for k, v in q['metrics'].items():
            print(f"    - {k:<12}: {v}")
            
    if q['technical_reasons']:
        print("  Technical Issues:")
        for reason in q['technical_reasons']:
            print(f"    - {reason}")

    print("-" * 80)
    print("[PREDICTION MODEL RESULTS (RAW SCORES)]")
    if item['context'] is not None:
        for model_key, res in item['context'].items():
            conf_pct = res['confidence'] * 100
            print(f"  Model '{model_key:<14}': Predicted Class Index {res['predicted_index']} ({conf_pct:.2f}% Confidence)")
    else:
        print("  [SKIPPED] Model inference bypassed because quality gate failed.")
        
    print("="*80)


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    selected_paths = collect_user_images()

    if not selected_paths:
        print("No photos were selected.")
        return

    print("Loading MobileNet models...")
    models = load_all_models(
        MODEL_CONFIG_MOBILENET,
        models_dir=r"I:\EEE\FYP_EE405\Contextual_Study\Context_MobileNet_550\results",
        device="cpu"
    )
    context_predictor = ImagePredictor(models)
    quality_assessor = QualityAssessor()

    batch_results = []

    print("\nProcessing images and evaluating metrics...")
    for path in selected_paths:
        filename = os.path.basename(path)
        
        # 1. Quality Gate Check
        q_res = quality_assessor.process(path)

        # 2. Context Prediction Pass (Triggered ONLY if Quality is ACCEPTED)
        if q_res['status'] == "ACCEPTED":
            c_res = context_predictor.predict(path)
        else:
            c_res = None

        item_data = {
            "filename": filename,
            "path": path,
            "quality": q_res,
            "context": c_res
        }

        # Print detailed technical metrics & raw predictions to terminal
        print_terminal_logs(item_data)

        batch_results.append(item_data)

    # Launch User-Friendly Subplot Dashboard
    dashboard = RowSubplotDashboard(batch_results)
    dashboard.show()


if __name__ == "__main__":
    main()