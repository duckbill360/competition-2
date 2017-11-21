# Functions for the competition 2.
import numpy as np

# hyperparameters
batch_size = 16
img_width = 500
img_height = 300
num_classes = 21


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


# Function from regression output to bounding box
def reg_to_bbox(reg, box):
    bbox_width = box[2] - box[0] + 1.0
    bbox_height = box[3] - box[1] + 1.0
    bbox_ctr_x = box[0] + 0.5 * bbox_width
    bbox_ctr_y = box[1] + 0.5 * bbox_height

    out_ctr_x = reg[0] * bbox_width + bbox_ctr_x
    out_ctr_y = reg[1] * bbox_height + bbox_ctr_y

    out_width = bbox_width * 10**reg[2]
    out_height = bbox_height * 10**reg[3]

    return np.array([
        max(0, out_ctr_x - 0.5 * out_width),
        max(0, out_ctr_y - 0.5 * out_height),
        min(img_width, out_ctr_x + 0.5 * out_width),
        min(img_height, out_ctr_y + 0.5 * out_height)
    ])




