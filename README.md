#  Drone Footage Object Detection
**AIML_012 | Problem Statement 3**

A deep learning pipeline for detecting, tracking, and alerting on objects in aerial drone footage. Built on the YOLO architecture and trained on the VisDrone dataset, the system supports general object detection, prompt-based detection, night-time inference, and a smart restricted-zone alert system — all accessible via a Streamlit web dashboard.

🔗 **Live App:** [dashboarddronefootageobjectdetection.streamlit.app](https://dashboarddronefootageobjectdetection.streamlit.app/)
🔗 **Dashboard Repo:** [AIML_012_PS_3_Dashboard](https://github.com/Shriya-1807/AIML_012_PS_3_Dashboard)

---

##  Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Overview](#model-overview)
- [How to Use](#how-to-use)
- [Results](#results)

---

##  Features

- **General Object Detection** — Detects cars, pedestrians, trucks, and more from aerial perspectives
- **Night-time Detection** — Robust performance in low-light drone footage
- **Prompt-Based Detection** — User-specified text prompts filter detection to only named object classes (e.g. *"yellow car, black car"*)
- **Smart Alert System** — Flags persons detected within a user-defined restricted zone and triggers an alarm
- **Video & Image Support** — Works on both static drone images and video sequences

---

##  Project Structure

```
AIML_012_PS_3/
│
├── Bonus Features/
│   └── Alert_system (2).ipynb       # Smart restricted-zone alert system
│
├── Inference/
│   ├── Text_prompt_vid_inference.ipynb       # Prompt-based detection on video
│   ├── text_prompt_inference_on_ima...ipynb  # Prompt-based detection on images
│   └── yolo_SAHI_inference (1).ipynb         # YOLO + SAHI inference pipeline
│
├── Results/
│   ├── BoxF1_curve.png              # F1 score curve
│   ├── BoxPR_curve.png              # Precision-Recall curve
│   ├── BoxP_curve.png               # Precision curve
│   ├── BoxR_curve.png               # Recall curve
│   ├── labels.jpg                   # Dataset label distribution
│   └── results (1).png              # Training results summary
│
├── Test/
│   └── test_evaluation.ipynb        # Model evaluation on test set
│
├── Training/
│   └── yolov8s_worldv2 (1).ipynb    # YOLOv8 model training notebook
│
├── reports/
│   ├── AIML-012_EndEval_FINAL_REP...pdf    # End evaluation report
│   ├── AIML-012_MidEval_Report.pdf         # Mid evaluation report
│   └── MidEval_AI-ML-012-PS3 (1).pptx     # Mid evaluation presentation
│
└── README.md
```

---

##  Dataset

The [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) is used — a benchmark specifically designed for aerial vision tasks.

**For Images:** Organized into `train/`, `val/`, and `test/` splits. Each split contains raw images alongside corresponding annotation files.

**For Videos:** Grouped into sequence folders, each containing video subfolders with extracted frame files. Annotation folders provide bounding box and class label data for every frame.

---

##  Model Overview

The detection system is built on the **YOLOv8 / YOLOv8-World** architecture, optimized for aerial perspectives and small, fast-moving targets. The pipeline covers:

1. Parsing and converting VisDrone annotations to YOLO format
2. Training on both image and video data (`Training/`)
3. Evaluation on a held-out test set (`Test/`)
4. Inference — standard, SAHI-enhanced, and text-prompt-based (`Inference/`)
5. A bonus alert module for restricted zone monitoring (`Bonus Features/`)

---

##  How to Use

### Prerequisites

```bash
pip install ultralytics streamlit torch torchvision opencv-python pillow
```

> Python 3.8+ recommended. A GPU with CUDA support is strongly advised for training and video inference.

---

### 1. Training the Model

Open and run `Training/yolov8s_worldv2 (1).ipynb`.

- Set the path to your VisDrone dataset at the top of the notebook
- Adjust hyperparameters (epochs, batch size, image size) as needed
- The trained model weights (`.pt` file) will be saved to your run directory

---

### 2. Evaluating on the Test Set

Open `Test/test_evaluation.ipynb`.

- Update the `model_path` variable to point to your trained weights
- Update the `data_path` variable to your VisDrone test split
- Run all cells to generate evaluation metrics and visualizations

---

### 3. Running Inference

#### Standard / SAHI Inference (Images & Video)
Open `Inference/yolo_SAHI_inference (1).ipynb`.
- Set your model path and input file path
- SAHI (Slicing Aided Hyper Inference) improves detection of small objects

#### Prompt-Based Detection on Images
Open `Inference/text_prompt_inference_on_ima...ipynb`.
- Enter your text prompt (e.g. `"pedestrian, bicycle"`) in the designated cell
- Only objects matching the prompt will be detected and displayed

#### Prompt-Based Detection on Video
Open `Inference/Text_prompt_vid_inference.ipynb`.
- Set the input video path and your text prompt
- Outputs an annotated video with only the prompted classes tracked

---

### 4. Smart Alert System

Open `Bonus Features/Alert_system (2).ipynb`.
- Load a drone image or video
- Define your restricted zone by specifying the **top-left** and **bottom-right** coordinates of the bounding box
- The system will detect persons within that zone and trigger an alert

---

### 5. Streamlit Dashboard

The easiest way to use all features interactively:

```bash
# Clone the dashboard repo
git clone https://github.com/Shriya-1807/AIML_012_PS_3_Dashboard
cd AIML_012_PS_3_Dashboard

# Install requirements
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Or access the hosted version directly: [dashboarddronefootageobjectdetection.streamlit.app](https://dashboarddronefootageobjectdetection.streamlit.app/)

---

##  Results

Training metrics (stored in `Results/`):

| Metric | Curve |
|---|---|
| F1 Score | `BoxF1_curve.png` |
| Precision-Recall | `BoxPR_curve.png` |
| Precision | `BoxP_curve.png` |
| Recall | `BoxR_curve.png` |

### General Object Detection
![General Object Detection](https://github.com/user-attachments/assets/7ce148ae-4716-4e38-8da5-4b3acd6fe336)

### Pedestrian Detection
![Pedestrian Detection](https://github.com/user-attachments/assets/d1b4459b-2d71-4028-bc67-f56c5f955271)

### Night-time Detection
https://github.com/user-attachments/assets/ae64f0b0-ec3d-4ad4-bdc1-1f64f8bfc998

### Prompt-Based Detection
Text prompt: *"yellow car, black car, white car"*
![Prompt Based Detection](https://github.com/user-attachments/assets/fe1b88b9-ead6-441e-86ba-e2c91c66c487)

Text prompt on video: *"red car"*
https://github.com/user-attachments/assets/19cacd9e-5dac-47a4-a3a5-29d0c3043a06

### Smart Alert System
![Alert System](https://github.com/user-attachments/assets/1ad17cc5-44a2-4b20-9bd6-7671a2d0dba3)

---

##  Reports

Full project documentation is available in the `reports/` folder:
- `AIML-012_EndEval_FINAL_REP...pdf` — End-semester evaluation report
- `AIML-012_MidEval_Report.pdf` — Mid-semester evaluation report
- `MidEval_AI-ML-012-PS3 (1).pptx` — Mid-semester presentation slides
