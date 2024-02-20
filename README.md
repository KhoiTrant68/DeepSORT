# DeepSORT
Re-implementation of Deep SORT.


## Futher improvement direction  
- Train detector on specific dataset rather than the official one.
- Retrain REID model on pedestrain dataset for better performance.
- Replace YOLOv3 detector with advanced ones.

**Any contributions to this repository is welcome!**


## Introduction
This is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN , and the original source code is [HERE](https://github.com/nwojke/deep_sort).  
However in original code, the CNN model is implemented with tensorflow, which I'm not familier with. SO I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

## Dependencies
- numpy
- scipy
- opencv-python
- sklearn
- torch
- torchvision 
- pillow
- vizer
- edict

## Quick Start
0. Check all dependencies installed
```bash
pip install -r requirements.txt
```

1. Clone this repository
```
git clone git@github.com:KhoiTrant/DeepSORT.git
```

2. Download YOLOv3 parameters
```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

3. Download deepsort parameters ckpt.t7
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

4. Compile nms module
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```


5. Run demo
```
usage: python yolov3_deepsort.py VIDEO_PATH
                                [--help]
                                [--frame_interval FRAME_INTERVAL]
                                [--config_detection CONFIG_DETECTION]
                                [--config_deepsort CONFIG_DEEPSORT]
                                [--display]
                                [--display_width DISPLAY_WIDTH]
                                [--display_height DISPLAY_HEIGHT]
                                [--save_path SAVE_PATH]          
                                [--cpu]          

# yolov3 + deepsort
python yolov3_deepsort.py [VIDEO_PATH]

# yolov3_tiny + deepsort
python yolov3_deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3_tiny.yaml

# yolov3 + deepsort on webcam
python3 yolov3_deepsort.py /dev/video0 --camera 0

# yolov3_tiny + deepsort on webcam
python3 yolov3_deepsort.py /dev/video0 --config_detection ./configs/yolov3_tiny.yaml --camera 0
```
Use `--display` to enable display.  
Results will be saved to `./output/results.avi` and `./output/results.txt`.


## Training the RE-ID model
The original model used in paper is in original_model.py, and its parameter here [original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).  

To train the model, first you need download [Market1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset .  

Then you can try [train.py](deep_sort/deep/train.py) to train your own parameter and evaluate it using [test.py](deep_sort/deep/test.py) and [evaluation.py](deep_sort/deep/evalution.py).
![train.jpg](deep_sort/deep/train.jpg)

## Demo videos and images
[demo.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
[demo2.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)


## References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- code: [ModelBunker/Deep-SORT-PyTorch](https://github.com/ModelBunker/Deep-SORT-PyTorch)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
