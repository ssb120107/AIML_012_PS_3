# AIML_012_PS_3
drone footage object detection

this project focuses on detecting objects in the envirnoment using drone footage.

The VisDrone dataset,which is organized in a highly useful format,serves as the starting point. it contains directories for training,validation and testing, each of which includes separate folders for images and annotations.This structure is primarily desgined for images and their corresponding annotations.
For videos,the format differs slightly:there are sequence folders containing multiple video files,with each sequence accompained by an annotation folder.the annotation folder includes the annotations which provides the dimensions of the bounding boxes required to enclose objects during detection.
Subsequently,we converted the VisDrone annotations into the yolo format,which is necessary for training the yolo model.

this is what we got after training our model--

![readme img](https://github.com/user-attachments/assets/22e7aa79-9f34-449c-be5d-40b358ec0c66)



https://github.com/user-attachments/assets/ae64f0b0-ec3d-4ad4-bdc1-1f64f8bfc998



https://github.com/user-attachments/assets/19cacd9e-5dac-47a4-a3a5-29d0c3043a06

![1000163037](https://github.com/user-attachments/assets/7ce148ae-4716-4e38-8da5-4b3acd6fe336)
![1000163039](https://github.com/user-attachments/assets/d1b4459b-2d71-4028-bc67-f56c5f955271)
![1000163081](https://github.com/user-attachments/assets/8453acd7-0f4f-41f7-bf1d-f9219c3f80c6)



