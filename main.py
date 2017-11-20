# CNN for Object Detection

# The classes ordered by index
classes = [
    '__background__',  # always index 0
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

########################################################################################################################
###### Preprocessing ######

# 1. Prepare bounding boxes for their class and regression data

# In object detection, we have two goals. One is to detect what the objects are, the other is to detect where they are.
# The first one is simply a classification problem. However, the second one can be seem as a regression problem.
# Assume we have some box proposals(roi, region of interst) after the machine see this region of the image.
# The machine should know how to move and resize the box proposal to output bounding box, which boxes the object.
import pandas as pd
import numpy as np

np.set_printoptions(edgeitems=10000)
np.core.arrayprint._line_width = 10000

df_ = pd.read_pickle('./dataset/train_data.pkl')
print(df_.head())


# We can prepare our training data by transforming coordinates [x1, y1, x2, y2]
# into [delta_x, delta_y, log(delta_w), log(delta_h)].


def bbox_transform(ex_rois, gt_rois):

    ex_widths = ex_rois[2] - ex_rois[0] + 1.0
    ex_heights = ex_rois[3] - ex_rois[1] + 1.0
    ex_ctr_x = ex_rois[0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[1] + 0.5 * ex_heights

    gt_widths = gt_rois[2] - gt_rois[0] + 1.0
    gt_heights = gt_rois[3] - gt_rois[1] + 1.0
    gt_ctr_x = gt_rois[0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.array([targets_dx, targets_dy, targets_dw, targets_dh])
    return targets


# Here we also resize the images into fixed 500*300 since the size of images are not the same
from PIL import Image

width = 500
height = 300
boxes_resize = df_['boxes'].copy()

for img in range(len(boxes_resize)):
    imgage = Image.open("./dataset/JPEGImages/" + df_['image_name'][img])
    w = imgage.size[0]
    h = imgage.size[1]
    boxes = boxes_resize[img]

    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (width / w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (height / h)
    boxes_resize[img] = np.array([df_['gt_classes'][img][0]] +
                                 bbox_transform(np.array([0, 0, width - 1, height - 1]), boxes[0]).tolist())

df_['one_gt'] = boxes_resize
print(df_.head())


#
class_count = [300 for i in range(21)]
df_select = df_.copy()
for img in range(len(df_select)):
    if class_count[int(df_select['one_gt'][img][0])] > 0:
        class_count[int(df_select['one_gt'][img][0])] -= 1
    else:
        df_select = df_select.drop(img)

df_select.reset_index(drop=True)
print(class_count)

df_.to_pickle('./dataset/data_train_one.pkl')


########################################################################################################################
###### Hyperparameters ######
batch_size = 16
img_width = 500
img_height = 300
num_classes = 21


###### Data loader ######
# In the following, we will introduce how we load data using tensorflow api.
import tensorflow as tf
import random
from tensorflow.contrib.data import Dataset, Iterator


# Split training data into train/valid sets
def _train_valid_split(df, valid_ratio):
    valid_random = np.random.rand(len(df)) < valid_ratio
    return df[~valid_random].reset_index(drop=True), df[valid_random].reset_index(drop=True)


df = pd.read_pickle('./dataset/data_train_one.pkl')
valid_ratio = 0.1
df_train, df_valid = _train_valid_split(df, valid_ratio)

# Define data_generator
# For each image, we generate an image array and its name. As a generator for tf.contrib.data.dataset to use.

def data_generator(image_name):
    file_path = './dataset/JPEGImages/'
    img_file = tf.read_file(file_path + image_name)

    img = tf.image.decode_image(img_file, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img.set_shape([None, None, 3])
    img = tf.image.resize_images(img, size=[img_width, img_height])

    return img, image_name


tf.reset_default_graph()


