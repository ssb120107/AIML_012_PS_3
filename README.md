# AIML_012_PS_3
# Drone Footage Object Detection

This project focuses on detecting and tracking objects in dynamic environments using drone footage. It leverages deep learning models (YOLO) alongside the VisDrone dataset for robust aerial object recognition in images and videos, including prompt-based detection features.

### Getting Started
  1. Clone the respository to your local machine.<br>
  2. Open the notebooks in your environment,you can use google colab or VS code.<br>
  3. follow the notebook to train and test the models.<br>

### Dataset
Visdrone dataset [(https://github.com/VisDrone/VisDrone-Dataset)] is used, specially designed for aerial vision tasks.<br>
For Images-<br>
  it is organized into folders for training, validation and testing.Each folder contains images    and matching annotation files.<br>
For Videos-<br>
  it is grouped into sequence folders, each containing multiple video files and their
  corresponding annotation folders provide bounding box data for every detected object in each 
  frame.<br>

### Model Overview
our detection system is based on the YOLO(you only look once) architecture, optimized for aerial perspectives and small, moving targets. the pipeline includes-<br>
  1. Parsing and converting visdrone annotations<br>
  2. training and evaluating on both images and videos<br>
  3. Prompt based used defined detection functionality<br>

### General Object Detectionn
objects such as cars, pedestrians, truck are accurately detected in drone images:

![General Object Detection](https://github.com/user-attachments/assets/7ce148ae-4716-4e38-8da5-4b3acd6fe336)

high accuracy pedestrian identification in complex scenes:

![Pedestrian Detection](https://github.com/user-attachments/assets/d1b4459b-2d71-4028-bc67-f56c5f955271)

### Nighttime Detection:
robust detection performance during night scenes :

https://github.com/user-attachments/assets/ae64f0b0-ec3d-4ad4-bdc1-1f64f8bfc998

### Prompt-Based Detection

detect and count only specified objects via text prompt 
here prompt was yellow,black and white cars:

![Prompt Based Detection](https://github.com/user-attachments/assets/8453acd7-0f4f-41f7-bf1d-f9219c3f80c6)

Example of prompt based detecting and tracking red cars in video 
**Watch Example Video**

https://github.com/user-attachments/assets/19cacd9e-5dac-47a4-a3a5-29d0c3043a06

![1000163467](https://github.com/user-attachments/assets/fe1b88b9-ead6-441e-86ba-e2c91c66c487)

















