import os
from settings import utils, LIB_DIR, COCO_MODEL_PATH

if __name__ == '__main__':

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    print('model weights loaded')
