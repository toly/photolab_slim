import sys
import os
from os.path import dirname

PROJECT_DIR = os.path.join(dirname(dirname(os.path.abspath(__file__))))
LIB_DIR = os.path.join(PROJECT_DIR, 'lib')
FILES_DIR = os.path.join(PROJECT_DIR, 'files')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MRCNN_PATH = os.path.join(LIB_DIR, 'Mask_RCNN')
COCO_PATH = os.path.join(MRCNN_PATH, 'samples/coco')
COCO_MODEL_PATH = os.path.join(LIB_DIR, "mask_rcnn_coco.h5")

sys.path.append(MRCNN_PATH)
sys.path.append(COCO_PATH)

CLASS_NAMES = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

from mrcnn import utils
import coco


# class InferenceConfig(coco.CocoConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
#
# config = InferenceConfig()
# config.display()
