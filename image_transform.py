'''
https://stackoverflow.com/questions/49163935/save-jaw-only-as-image-with-dlib-facial-landmark-detection-and-the-rest-to-be-tr

pip3 install imutils

sudo apt-get update
sudo apt-get install libtbb2

'''
import numpy as np

# def person_cut(image, ):


# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
# # %%
# os.path.isfile(os.path.join(path_data, 'Faces', 'shape_predictor_68_face_landmarks.dat'))
#
# # %%
# import glob
# for i in glob.glob(os.path.join(path_data, 'Faces', '*')):
#     print(i)
#
# print(os.getcwd())
# # %%
path_data = 'data'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(path_data, 'Faces', 'shape_predictor_68_face_landmarks.dat'))

# load image
img = cv2.imread(os.path.join(path_data, 'Faces', 'thegovernator.png'))
h, w, ch = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# add an alpha channel to image
b,g,r = cv2.split(img);
a = np.ones((h,w,1), np.uint8) * 255
img = cv2.merge((b, g, r, a))
## %%
# detect face
rects = detector(gray,1)
roi = rects[0] # region of interest
shape = predictor(gray, roi)
shape = face_utils.shape_to_np(shape)
# extract jawline
jawline = shape[0:17]
top = min(jawline[:,1])
bottom = max(jawline[:,1])
# extend contour for masking
jawline = np.append(jawline, [ w-1, jawline[-1][1] ]).reshape(-1, 2)
jawline = np.append(jawline, [ w-1, h-1 ]).reshape(-1, 2)
jawline = np.append(jawline, [ 0, h-1 ]).reshape(-1, 2)
jawline = np.append(jawline, [ 0, jawline[0][1] ]).reshape(-1, 2)
contours = [ jawline ]
# generate mask
mask = np.ones((h,w,1), np.uint8) * 255 # times 255 to make mask 'showable'
cv2.drawContours(mask, contours, -1, 0, -1) # remove below jawline
# apply to image
result = cv2.bitwise_and(img, img, mask = mask)
result = result[top:bottom, roi.left():roi.left()+roi.width()] # crop ROI
cv2.imwrite(os.path.join(path_data, 'Faces', 'result.png'), result)
# %%
# cv2.imshow(('masked image', result)
# %%
# from PIL import Image
# img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
# im_pil = Image.fromarray(img)
# # %%
# from matplotlib import pyplot as plt
# plt.imshow(im_pil)
# plt.show()

