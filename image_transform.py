'''
https://stackoverflow.com/questions/49163935/save-jaw-only-as-image-with-dlib-facial-landmark-detection-and-the-rest-to-be-tr

pip3 install imutils

sudo apt-get update
sudo apt-get install libtbb2

sudo pip3 install imageio

'''
import numpy as np

# def person_cut(image, ):


# # import the necessary packages
# from imutils import face_utils
# import numpy as np
# import argparse
# import imutils
# import dlib
# import cv2
# import os

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils, dlib, cv2, os
# import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from PIL import Image

def resize(arr_image, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    # ratio_w = width / arr_image.width
    # ratio_h = height / arr_image.height
    arr_image_width = arr_image.shape[1]#.width
    arr_image_height = arr_image.shape[0]#.height

    ratio_w = width / arr_image_width
    ratio_h = height / arr_image_height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * arr_image_height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * arr_image_width)
        resize_height = height

    # image_resize = arr_image.resize((resize_width, resize_height), Image.ANTIALIAS)
    image_resize = Image.fromarray(arr_image).resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)

    # return background.convert('RGB')
    return np.array(background.convert('RGBA'))


def arr_image_add_alpha(arr_image:np.ndarray)->np.ndarray:
    '''add an alpha channel to image'''
    h, w, ch = arr_image.shape
    b,g,r = cv2.split(arr_image);
    a = np.ones((h, w, 1), np.uint8) * 255

    return cv2.merge((b, g, r, a))


def face_get(path_file_in:str,
             path_file_result:str,
             path_file_shape_predictor_68_face_landmarks:str='shape_predictor_68_face_landmarks.data',
             # coordinates_face:tuple=None
             ):

    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(os.path.join(path_data, 'Faces', 'shape_predictor_68_face_landmarks.dat'))

    assert os.path.isfile(path_file_shape_predictor_68_face_landmarks), 'File "{0}" not exist'.format(path_file_shape_predictor_68_face_landmarks)
    predictor = dlib.shape_predictor(path_file_shape_predictor_68_face_landmarks)

    # load image
    assert os.path.isfile(path_file_in), 'File "{0}" not exist'.format(path_file_in)
    img_in = cv2.imread(path_file_in)

    h, w, ch = img_in.shape
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # add an alpha channel to image
    # b,g,r = cv2.split(img_in);
    # a = np.ones((h,w,1), np.uint8) * 255
    # img = cv2.merge((b, g, r, a))
    img = arr_image_add_alpha(img_in)

    # detect face
    rects = detector(gray,1)
    if len(rects) == 0:
        # return None, None
        if img_in.shape[-1] == 3:
            img_in = arr_image_add_alpha(img_in)
            y, x, _ = np.where(img_in[:,:, :-1]>0)
            y_from, y_to = (y.min(), y.max()+1)
            x_from, x_to = (x.min(), x.max()+1)
            img_in = img_in[y_from: y_to, x_from: x_to, :]
            # img_in[:,:, 3] =
            fff =111
            # img_in[np.where(img_in[:, :, :-1].sum(axis=2) == 0), -1] = 0
            x = np.where(img_in[:, :, :-1].sum(axis=2) == 0)
            img_in[x[0], x[1], -1] = 0

        return img_in, (0, 0)

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
    coordinates_face = (top, roi.left())
    # apply to image
    result = cv2.bitwise_and(img, img, mask = mask)
    result = result[top:bottom, roi.left():roi.left()+roi.width()] # crop ROI
    if not path_file_result is None:
        cv2.imwrite(path_file_result, result)

    return result, coordinates_face


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
    # dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

'''
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
'''

def folder_frame_prepare(path_frame:str):

    # path_frame = os.path.join(path_data, 'Faces', 'frames')
    os.makedirs(path_frame, exist_ok=True)
    for i in os.listdir(path_frame):
        os.remove(os.path.join(path_frame, i))

def array_past(arr_back:np.ndarray, arr_insert:np.ndarray, coordinate_past:tuple):

    y_offset, x_offset = coordinate_past
    y1, y2 = y_offset, y_offset + arr_insert.shape[0]
    x1, x2 = x_offset, x_offset + arr_insert.shape[1]

    alpha_s = arr_insert[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        arr_back[y1:y2, x1:x2, c] = (alpha_s * arr_insert[:, :, c] +
                                     alpha_l * arr_back[y1:y2, x1:x2, c])


    return arr_back

def frames_create(path_file_in:str, path_frames:str,
                  path_file_back:str=None,
                  coordinate_past:tuple=None,
                  step_k_shape_0:float = 0.002,
                  step_k_shape_1:float = 0.002,
                  range_k_shape_0:tuple = (0.0, 0.05),
                  range_k_shape_1:tuple = (0.01, 0.05),
                  seq:list = [], cnt:int = 0,
                  asceding:bool=True,
                  verbose: int = 0)->int:

    # step_k_shape_0 = 0.002
    # step_k_shape_1 = 0.002
    # range_k_shape_0 = (0.0, 0.05)
    # range_k_shape_1 = (0.01, 0.05)
    # im_merge = cv2.imread(os.path.join(path_data, 'Faces', 'result.png'), -1)

    assert os.path.isfile(path_file_in), 'File "{0}" not exist'.format(path_file_in)

    im_merge = cv2.imread(path_file_in, -1)

    if not path_file_back is None:
        assert os.path.isfile(path_file_back), 'File "{0}" not exist'.format(path_file_back)
        im_back = cv2.imread(path_file_back, -1)
        assert not coordinate_past is None and len(coordinate_past)==2, 'coordinate_past "{0}" is bad'.format(coordinate_past)


    def _range_k_shape(range_k_shape, step_k_shape, asceding)->tuple:
        if asceding:
            return min(range_k_shape), \
                    ((1+ np.floor(max(range_k_shape)/abs(step_k_shape)))*step_k_shape), \
                    abs(step_k_shape)
        else:
            return max(range_k_shape), \
                    ((-1+ np.floor(min(range_k_shape)/abs(step_k_shape)))*step_k_shape), \
                    -abs(step_k_shape)



    k_shape_0_from, k_shape_0_to, k_shape_0_step =\
        _range_k_shape(range_k_shape_0, step_k_shape_0, asceding)
    for i, k_shape_0 in enumerate(
            np.arange(k_shape_0_from, k_shape_0_to, k_shape_0_step)):

        k_shape_1_from, k_shape_1_to, k_shape_1_step = \
            _range_k_shape(range_k_shape_1, step_k_shape_1, asceding and i % 2 == 0)

        # print(i, cnt, 'k_shape_0', k_shape_0)
        for k_shape_1 in np.arange(k_shape_1_from, k_shape_1_to, k_shape_1_step):
            # print(i, cnt, 'k_shape_1', k_shape_1)
            # Apply transformation on image
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2,
                                           im_merge.shape[1] * k_shape_1,
                                           im_merge.shape[1] * k_shape_0)
            path_file = 'result_non_linear_{0:04d}_{1:.4f}_{2:.4f}.png'.format(cnt, k_shape_0, k_shape_1)

            if not path_file_back is None:

                im_back = array_past(im_back, im_merge_t, coordinate_past)

                # im_back.paste(im_merge_t, coordinate_past, im_merge_t)
                cv2.imwrite(os.path.join(path_frames, path_file), im_back)
                seq.append(im_back)
            else:
                cv2.imwrite(os.path.join(path_frames, path_file), im_merge_t)
                seq.append(im_merge_t)

            if verbose: print('Frame created :', path_file)
            cnt += 1

    return cnt

path_data = 'data'
# path_file_in = os.path.join(path_data, 'Faces', 'thegovernator.png')
# path_file_in = os.path.join(path_data, 'Faces', 'pic_d.jpg')
# path_file_in = os.path.join(path_data, 'Faces', 'jury_2.jpg')
# path_file_in = os.path.join(path_data, 'picture_6.jpg')
path_file_in = os.path.join(path_data, 'Faces', 'Dmitry.jpg')
path_file_result = os.path.join(path_data, 'Faces', 'result.png')
path_file_shape_predictor_68_face_landmarks = os.path.join(path_data, 'Faces', 'shape_predictor_68_face_landmarks.dat')
path_frames = os.path.join(path_data, 'Faces', 'frames')

_, coordinates_face = face_get(path_file_in, path_file_result, path_file_shape_predictor_68_face_landmarks)
folder_frame_prepare(path_frames)
seq = []
# frames_create(path_file_in,
cnt = frames_create(path_file_result,
                    path_frames,
                    path_file_in, coordinates_face,
                    step_k_shape_0 = 0.0025,
                    step_k_shape_1 = 0.0025,
                    range_k_shape_0 = (0.035, 0.05),
                    range_k_shape_1 = (0.04, 0.05),
                    seq = seq, cnt = 0,
                    asceding=True,
                    verbose = 1)

# os._exit(0)
# %%
# # import cv2
# # import os
# #
# # vvw           =   cv2.VideoWriter(os.path.join(path_frame, 'mymovie.avi'),
# #                                   cv2.VideoWriter_fourcc('X','V','I','D'),24,
# #                                   (640,480))
# # frameslist    =   os.listdir(path_frame)
# # howmanyframes =   len(frameslist)
# # print('Frames count: '+str(howmanyframes)) #just for debugging
# #
# # for i in range(0,howmanyframes):
# #     print(i)
# #     theframe = cv2.imread(os.path.join(path_frame, frameslist[i]))
# #     vvw.write(theframe)
#
#
# # %%
# import imageio
# if os.path.isfile(os.path.join(path_frames, 'movie.gif')):
#     os.remove(os.path.join(path_frames, 'movie.gif'))
# # images = []
# # for filename in filenames:
# #     images.append(imageio.imread(filename))
# # imageio.mimsave('/path/to/movie.gif', images)
# imageio.mimsave(os.path.join(path_frames, 'movie.gif'), seq)
# %%
path_file_back = os.path.join(path_data, 'Faces', 'thegovernator.png')
path_file_in = os.path.join(path_data, 'other', 'bug_5.png')
# path_file_in = os.path.join(path_data, 'other', 'gorilla_2.png')

img_in = cv2.imread(path_file_in)
img_back = cv2.imread(path_file_back)

in_result, in_coordinates_face = face_get(path_file_in,
                                      path_file_result=None,
                                      path_file_shape_predictor_68_face_landmarks=path_file_shape_predictor_68_face_landmarks,
                                      )
back_result, back_coordinates_face = face_get(path_file_back,
                                              path_file_result=None,
                                              path_file_shape_predictor_68_face_landmarks=path_file_shape_predictor_68_face_landmarks,
                                              )

in_result_resized = resize(in_result, width=back_result.shape[1], height=back_result.shape[1])
print(in_result.shape, back_result.shape, in_result_resized.shape)

# # %%
# import glob
# for filename in glob.glob(os.path.join(path_data, '*')):
#     print(filename)

im_back_inplaced = array_past(img_back, in_result_resized, back_coordinates_face)
img_back.shape, in_result_resized.shape

path_file_im_back_inplaced = os.path.join(path_data, 'Faces', 'im_back_inplaced_{0}_{1}.png'\
                                          .format(os.path.basename(path_file_back), os.path.basename(path_file_in)))
cv2.imwrite(path_file_im_back_inplaced, im_back_inplaced)

