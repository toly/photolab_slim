'''
https://stackoverflow.com/questions/49163935/save-jaw-only-as-image-with-dlib-facial-landmark-detection-and-the-rest-to-be-tr

pip3 install imutils

sudo apt-get update
sudo apt-get install libtbb2

sudo pip3 install imageio

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

top = rects[0].top()
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
import os

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

# %%
import numpy as np
import pandas as pd
import cv2, os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

path_data = 'data'

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


im_merge = cv2.imread(os.path.join(path_data, 'Faces', 'result.png'), -1)

k_shape_0 = 0.05
k_shape_1 = 0.05

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2,
                               im_merge.shape[1] * k_shape_1,
                               im_merge.shape[1] * k_shape_0)

# # Apply transformation on image
# im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
# plt.figure(figsize = (16,14))
# plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
cv2.imwrite(os.path.join(path_data, 'Faces', 'result_non_linear.png'), im_merge_t)

## %%
path_frame = os.path.join(path_data, 'Faces', 'frames')
os.makedirs(path_frame, exist_ok=True)
for i in os.listdir(path_frame):
    os.remove(os.path.join(path_frame, i))

step_k_shape_0 = 0.002
step_k_shape_1 = 0.002
range_k_shape_0 = (0.0, 0.05)
range_k_shape_1 = (0.01, 0.05)
im_merge = cv2.imread(os.path.join(path_data, 'Faces', 'result.png'), -1)

cnt = 0
seq = []
for i, k_shape_0 in enumerate(
                    np.arange(range_k_shape_0[0],
                    range_k_shape_0[1]+step_k_shape_0,
                    step_k_shape_0)):

    if i % 2 == 0:
        range_k_shape_1_from, range_k_shape_1_to, step = range_k_shape_1[0], (range_k_shape_1[1] + step_k_shape_1), step_k_shape_1
    else:
        range_k_shape_1_from, range_k_shape_1_to, step = range_k_shape_1[1], (range_k_shape_1[1] - step_k_shape_1), -step_k_shape_1

    # print(i, cnt, 'k_shape_0', k_shape_0)
    for k_shape_1 in np.arange(range_k_shape_1_from, range_k_shape_1_to, step):
        # print(i, cnt, 'k_shape_1', k_shape_1)
        # Apply transformation on image
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2,
                                       im_merge.shape[1] * k_shape_1,
                                       im_merge.shape[1] * k_shape_0)
        path_file = 'result_non_linear_{0:05d}_{1:.4f}_{2:.4f}.png'.format(cnt, k_shape_0, k_shape_1)
        cv2.imwrite(os.path.join(path_frame, path_file), im_merge_t)
        seq.append(im_merge_t)
        print(path_file)
        cnt += 1

    #     if cnt > 3:
    #         break
    #     # break
    # # print('result_non_linear_{0}_{1:.2f}_{2:.2f}.png'.format(
    # #     cnt, k_shape_0, k_shape_1))
    # if cnt > 3:
    #     break
    # # break

# # %%
# import cv2
# import os
#
# vvw           =   cv2.VideoWriter(os.path.join(path_frame, 'mymovie.avi'),
#                                   cv2.VideoWriter_fourcc('X','V','I','D'),24,
#                                   (640,480))
# frameslist    =   os.listdir(path_frame)
# howmanyframes =   len(frameslist)
# print('Frames count: '+str(howmanyframes)) #just for debugging
#
# for i in range(0,howmanyframes):
#     print(i)
#     theframe = cv2.imread(os.path.join(path_frame, frameslist[i]))
#     vvw.write(theframe)
# %%
import imageio
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images)
imageio.mimsave(os.path.join(path_frame, 'movie.gif'), seq)
