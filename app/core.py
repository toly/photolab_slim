from app.settings import *

import numpy as np
import cv2 as cv
from PIL import Image

import mrcnn.model as modellib
import coco
import skimage


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


model = modellib.MaskRCNN(mode='inference', model_dir=LIB_DIR, config=config)


model.load_weights(COCO_MODEL_PATH, by_name=True)


def load_image(path):
    return skimage.io.imread(path)


def do_thin(image, ratio=0.85):
    results = model.detect([image], verbose=1)
    r = results[0]

    mask = r['masks'][:, :, 0]
    _, x1, _, x2 = r['rois'][0]

    dilate_wide = int((x2 - x1) * 0.05)
    mask = cv.dilate((mask * 255).astype(np.uint8), np.ones((dilate_wide, dilate_wide), np.uint8))

    image_4d = np.concatenate([image, mask[:, :, np.newaxis]], axis=2)
    image_alpha = Image.fromarray(image_4d.astype(np.uint8), mode='RGBA')
    resized_cut = image_alpha.resize((int(image_alpha.width * ratio), image_alpha.height), Image.ANTIALIAS)

    result_inpaint = cv.inpaint(image_4d[:, :, :3].astype(np.uint8), mask.astype(np.uint8) * 255, 5, cv.INPAINT_TELEA)
    result_inpaint = Image.fromarray(result_inpaint)

    new_x = int((x1 + x2) / 2 * (1 - ratio))
    result_inpaint.paste(resized_cut, (new_x, 0), resized_cut)

    return result_inpaint
