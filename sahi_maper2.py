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
