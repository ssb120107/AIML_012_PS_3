# AIML_012_PS_3
# Drone Footage Object Detection

This project focuses on detecting and tracking objects in dynamic environments using drone footage. It leverages deep learning models (YOLO) alongside the VisDrone dataset for robust aerial object recognition in images and videos, including prompt-based detection features.

### Dataset
Visdrone dataset [(https://github.com/VisDrone/VisDrone-Dataset)] is used, which is specially designed for aerial vision tasks.<br>
For Images-<br>
  the dataset is organized into folders for training, validation and testing.Each folder contains images and their corresponding annotation files.<br>
For Videos-<br>
  the dataset is grouped into sequence folders, each containing multiple video sub folders, each of which contains frame files of that video. Their
  corresponding annotation folders provide data for creating bounding boxes and categorizing every object to be detected in each 
  frame.<br>

### Model Overview
Our detection system is based on the YOLO(you only look once) architecture, optimized for aerial perspectives and small, moving targets. The pipeline includes-<br>
  1. Parsing and converting visdrone annotations<br>
  2. Training and evaluating on both- images, and videos<br>
  3. Text-prompt based detection functionality<br>

### General Object Detectionn
Objects such as cars, pedestrians, truck are accurately detected in drone images:

![General Object Detection](https://github.com/user-attachments/assets/7ce148ae-4716-4e38-8da5-4b3acd6fe336)

High accuracy pedestrian/ vehicles identification (pedestrians as visible in the ground) in complex scenes:

![Pedestrian Detection](https://github.com/user-attachments/assets/d1b4459b-2d71-4028-bc67-f56c5f955271)

### Nighttime Detection:
Robust detection performance during night scenes :

https://github.com/user-attachments/assets/ae64f0b0-ec3d-4ad4-bdc1-1f64f8bfc998

### Prompt-Based Detection

Detect and count only specified objects via text prompt 
Here, text-prompt enetered by user was: "yellow car, black car, white car":

![Prompt Based Detection](https://github.com/user-attachments/assets/fe1b88b9-ead6-441e-86ba-e2c91c66c487)

Sample output of prompt based detection and tracking. Text prompt entered for input video file: "red car"
**Watch Example Video**

https://github.com/user-attachments/assets/19cacd9e-5dac-47a4-a3a5-29d0c3043a06



















