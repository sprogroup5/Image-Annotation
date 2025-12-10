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
