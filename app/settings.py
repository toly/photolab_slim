import sys
import os
from os.path import dirname

PROJECT_DIR = os.path.join(dirname(dirname(os.path.abspath(__file__))))
LIB_DIR = os.path.join(PROJECT_DIR, 'lib')
FILES_DIR = os.path.join(PROJECT_DIR, 'files')
MRCNN_PATH = os.path.join(LIB_DIR, 'Mask_RCNN')
COCO_PATH = os.path.join(MRCNN_PATH, 'samples/coco')
COCO_MODEL_PATH = os.path.join(LIB_DIR, "mask_rcnn_coco.h5")

sys.path.append(MRCNN_PATH)
sys.path.append(COCO_PATH)

# from mrcnn import utils
# import coco


# class InferenceConfig(coco.CocoConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
#
# config = InferenceConfig()
# config.display()


