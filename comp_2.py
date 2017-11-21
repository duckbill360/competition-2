# Functions for the competition 2.
import numpy as np


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



