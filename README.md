# AIML_012_PS_3
drone footage object detection

this project focuses on detecting objects in the envirnoment using drone footage.

The VisDrone dataset,which is organized in a highly useful format,serves as the starting point. it contains directories for training,validation and testing, each of which includes separate folders for images and annotations.This structure is primarily desgined for images and their corresponding annotations.
For videos,the format differs slightly:there are sequence folders containing multiple video files,with each sequence accompained by an annotation folder.the annotation folder includes the annotations which provides the dimensions of the bounding boxes required to enclose objects during detection.
Subsequently,we converted the VisDrone annotations into the yolo format,which is necessary for training the yolo model.

this is what we got after training our model--


