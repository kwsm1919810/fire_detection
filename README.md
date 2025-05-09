# fire_detection
My codes for Scientific Writing and Presentation Skills (3729-080)
<h1 align="center"><span>YOLOv5 for Fire Detection</span></h1>

Fire detection task aims to identify fire or flame in a video and put a bounding box around it. This repo includes a demo on how to build a fire detector using YOLOv5. 

<p align="center">
  <img src="results/result.gif" />
</p>

## 🏋️ Training
```train.ipynb``` script is for training the model. 
- YOLOv5
```
python train.py --img 640 --batch 16 --epochs 3 --data ../fire.yaml --weights yolov5s.pt --workers 1
```

## 🌱 Inference

- YOLOv5
``` shell
python detect.py --weights runs/train/exp4/weights/best.pt --img 640 --conf 0.25 --source ../datasets/fire/val/images/
```

## ⏱️ Results
The following charts were produced after training YOLOv5s with input size 640x640 on the fire dataset for 10 epochs.

| Ground Truth | Prediction | 
| :-: | :-: |
| ![](results/val_batch2_labels_1.jpg) | ![](results/val_batch2_pred_1.jpg) |
| ![](results/val_batch2_labels_2.jpg) | ![](results/val_batch2_pred_2.jpg) | 

## 🔗 Reference
I borrowed and modified [YOLOv5-Custom-Training.ipynb](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) script for training YOLOv5 model on the fire dataset. For more information on training YOLOv5, please refer to its homepage.
* https://github.com/robmarkcole/fire-detection-from-images
* https://github.com/ultralytics/yolov5
* https://github.com/AlexeyAB/darknet