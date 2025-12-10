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
## **8. Continuous Model Improvement** 
- Review performance metrics frequently.
- Add more annotated images whenever performance plateaus.
- Retrain or fine-tune models.
- Re-evaluate using SAHI to confirm reliability and robustness.
## **9. Code Structure**

## **Notes** 
- Make sure your virtual environment is activated whenever running tools or training.
- Keep all dependencies updated to ensure compatibility.
- Organize your dataset consistently for best YOLO results.
