## Run
```bash
# download yolov3.weights from darknet: 
# https://pjreddie.com/media/files/yolov3.weights
CUDA_VISIBLE_DEVICES=0 python val.py \
--weights ./yolov3-v9.5.0.pt \
--data coco.yaml \
--imgsz 416 \
--task speed \
--darknet
```
