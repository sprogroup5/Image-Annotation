# Image Annotation and Object Detection Workflow: ARAMSAM + YOLO + SAHI
This documentation provides a complete end-to-end workflow for image annotation, **YOLO model training**, tiled inference evaluation with **SAHI**, and establishing a continuous improvement loop.

<img width="1140" height="450" alt="image" src="https://github.com/user-attachments/assets/427035c8-fa14-4e88-9daa-ba6cf1e8b16f" />

## **Prerequisites** 
-  Python **3.11 or newer** is required.-
-  It is recommended to use a **virtual environment (venv)** for clean dependency 
management. 
## **1. Clone the ARAMSAM Repository** 
**ARAMSAM** is an annotation tool that uses the Segment Anything Model (SAM) to accelerate mask and bounding-box creation. It allows you to generate high-quality labels much faster than manual tools, reducing annotation time.
Clone the official repository into your VSCode workspace or terminal: 
```bash 
git clone https://github.com/DerOehmer/ARAMSAM 
CD ARAMSAM 
```
## **2. Create and Activate a Virtual Environment** 
Create a virtual environment: 
```bash 
python3 -m venv venv 
``` 
Activate it: 

**macOS / Linux** 
```bash 
source venv/bin/activate 
``` 
**Windows** 
```bash 
venv\Scripts\activate 
``` 
--- 
## **3. Install Dependencies** 
Install all required Python packages: 
```bash 
pip install -r requirements.txt 
```
## **4. Install SAM (Segment Anything Model)** 
Download and install your preferred SAM version from: 
https://github.com/facebookresearch/sam2 

Place the downloaded `.pth` checkpoint files (e.g., `sam_vit_b_01ec64.pth`, 
`sam_vit_l_0b3195.pth`) into the correct directory as described in the repository.  
## **5. Start Annotating Images** 
With the environment fully set up, you can now begin annotating images according to your 
project needs using ARAMSAM. 
## **6. Train YOLO** 
Once you have created enough high-quality annotations: 
1. Install YOLO: 
```bash 
pip install ultralytics 
``` 
2. Train your YOLO model with settings that match your performance requirements. 
Always check your company's accuracy criteria to ensure the model meets expected 
standards.
```bash
from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")

    # Folder name for this new run
    folder_name = "yolo11m500epochsbest1280"

    # Training configuration
    model.train(
        data="faba.yaml",
        epochs=500,          
        imgsz=1280,        
        device=0,
        batch=16,            
        workers=5,
        project="runs/detect",
        name=folder_name,
        exist_ok=True,
        patience=20,          
        pretrained=True,
    )

    print(f"\n🎉 Training completed and saved in runs/detect/{folder_name}")


if __name__ == "__main__":
    main()
```
--- 
## **7. Evaluate the Model With SAHI** 
After YOLO achieves acceptable performance (e.g., **R² ≥ 0.8**, **MAPE < 15%**, or 
internal thresholds): 

Install SAHI for tiled inference: 
```bash 
pip install sahi 
``` 
Follow the o icial ultralytics SAHI guide: 
https://docs.ultralytics.com/guides/sahi-tiled-inference 
Use SAHI to test how well your model performs autonomously on large or complex images.
Example SAHI code to use (good performace proven):
```bash
from pathlib import Path
import csv

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

# =============================
# CONFIG
# =============================

MODEL_PATH = Path("runs/detect/yolo11m500epochsbest1280/weights/best.pt")
BIG_PLOTS_DIR = Path("mix78")

VIS_OUTPUT_DIR = Path("sahi_visuals")
CSV_OUTPUT_PATH = Path("plot_counts.csv")
GT_COUNTS_CSV = Path("ground_truth_counts.csv")

VIS_OUTPUT_DIR.mkdir(exist_ok=True)

# Load SAHI model
detection_model = UltralyticsDetectionModel(
    model_path=str(MODEL_PATH),
    confidence_threshold=0.15,
    device="cuda:0",
)


# =============================
# METRICS
# =============================

def compute_r2_and_mape(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return None, None

    mean_y = sum(y_true) / n

    # R²
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # MAPE
    ape = []
    for yt, yp in zip(y_true, y_pred):
        if yt != 0:
            ape.append(abs((yt - yp) / yt) * 100)

    mape = sum(ape) / len(ape) if ape else None

    return r2, mape


# =============================
# MAIN
# =============================

def main():
    # CSV header
    rows = [["image_name", "faba_bean_count", "weed_count", "total"]]

    # SAHI inference
    for img_path in sorted(BIG_PLOTS_DIR.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue

        print(f"\nProcessing {img_path.name} ...")

        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )

        faba = 0
        weed = 0

        for obj in result.object_prediction_list:
            cls_id = obj.category.id
            if cls_id == 0:
                faba += 1
            elif cls_id == 1:
                weed += 1

        total = faba + weed
        rows.append([img_path.name, faba, weed, total])

        print(f"   faba: {faba}  weed: {weed}  total: {total}")

        result.export_visuals(
            export_dir=str(VIS_OUTPUT_DIR),
            file_name=img_path.stem,
        )

    # Save prediction CSV
    with CSV_OUTPUT_PATH.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print("\nDone!")
    print(f"Prediction counts saved to: {CSV_OUTPUT_PATH}")
    print(f"Visuals saved in: {VIS_OUTPUT_DIR}")

    # ======================================
    # R² & MAPE
    # ======================================
    if GT_COUNTS_CSV.exists():
        print("\nGround truth file found, computing R² and MAPE for faba bean counts...")

        # Load ground truth
        gt_dict = {}
        with GT_COUNTS_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if "image_name" not in reader.fieldnames:
                raise ValueError("Ground truth CSV must contain an 'image_name' column")

            for row in reader:
                name = row["image_name"].strip()
                gt_dict[name] = int(row["faba_count"])

        # Load predictions
        pred_dict = {}
        with CSV_OUTPUT_PATH.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["image_name"].strip()
                pred_dict[name] = int(row["faba_bean_count"])

        # Align and compare only matching images
        y_true = []
        y_pred = []

        for name in gt_dict:
            if name in pred_dict:
                y_true.append(gt_dict[name])
                y_pred.append(pred_dict[name])
            else:
                print(f"WARNING: {name} missing from predictions, skipping...")

        r2, mape = compute_r2_and_mape(y_true, y_pred)

        print("\n===== FINAL METRICS =====")
        print(f"R²    = {r2:.4f}")
        print(f"MAPE  = {mape:.2f}%")
        print("=========================")


if __name__ == "__main__":
    main()
``` 
## **8. Continuous Model Improvement** 
- Review performance metrics frequently.
- Add more annotated images whenever performance plateaus.
- Retrain or fine-tune models.
- Re-evaluate using SAHI to confirm reliability and robustness.

## **Training Set Up**
Three YOLO11 model variants were tested systematically.
- **yolo11ncode.py** (lightweight, fast, low capacity)
- **yolo11mcode.py** (lightweight, fast, low capacity)
- **yolo11lcode.py** (high capacity, slower, heavy VRAM usage)

Across all experiments, **YOLO11m** proved to be the most effective model for detecting faba beans and weeds. **YOLO11n** was fast but lacked accuracy, struggling with small weeds. **YOLO11l** provided slightly better accuracy but required significantly more computation without offering meaningful improvements. In contrast, **YOLO11m** trained at 1280 px resolution delivered the best balance between performance and efficiency, excellent class-level results (0.978 for faba beans and 0.917 for weeds).
This makes **YOLO11m** the recommended model for the project in terms of accuracy, stability, and resource usage.

## **Dataset Split and Data Handling**
To ensure reliable training and unbiased evaluation, the dataset was divided into:
- **50%** training
- **30%** validation
- **20%** test
  
## **Notes** 
- Make sure your virtual environment is activated whenever running tools or training.
- Keep all dependencies updated to ensure compatibility.
- Organize your dataset consistently for best YOLO results.**

