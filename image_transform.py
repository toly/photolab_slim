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

# %%

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True, help="Path to input image")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
#
# # load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
path_data = 'data'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(path_data, 'Faces', 'shape_predictor_68_face_landmarks.dat'))

# load image
image = cv2.imread(os.path.join(path_data, 'Faces', 'thegovernator.png'))
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    print(face_utils.FACIAL_LANDMARKS_IDXS.items())
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        print(" i = ", i, " j = ", j)
        # clone the original image so we can draw on it, then
        # display the name of the face part of the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # loop over the subset of facial landmarks, drawing the
        # specific face part using a red dots
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y+h,x:x+w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # show the particular face part
        # cv2.imshow("ROI", roi)
        cv2.imwrite(os.path.join(path_data, 'Faces', name + '.jpg'), roi)
        # cv2.imshow("Image", clone)
        cv2.imwrite(os.path.join(path_data, 'Faces', name + '_clone.jpg'), clone)
        # cv2.waitKey(0)

    # visualize all facial landmarks with a transparent overly
    # output = face_utils.visualize_facial_landmarks(image, shape)
    # cv2.waitKey(0)