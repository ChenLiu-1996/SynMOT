import os
from glob import glob

import cv2
import numpy as np


def load_image_series(mot_data_folder, seq, first_k=10):
    image_series = []
    if first_k is not None:
        num_images = first_k
    else:
        num_images = len(
            glob(os.path.join(mot_data_folder, '%s/img1/*.jpg' % seq)))

    for i in range(num_images):
        image = cv2.imread(
            os.path.join(mot_data_folder,
                         '{}/img1/{:06d}.jpg'.format(seq, i + 1)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_series.append(image)

    return image_series


def load_annotation_series(mot_data_folder, seq, first_k=10):
    annotation_series = []

    seq_path = os.path.join(mot_data_folder, seq)
    ann_path = os.path.join(seq_path, 'gt/gt.txt')
    anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

    ann_cnt = 0  # annotation count
    tid_curr = 0  # track id
    tid_last = -1

    for i in range(anns.shape[0]):
        frame_id = int(anns[i][0])
        if first_k is not None and frame_id > first_k:
            continue
        track_id = int(anns[i][1])
        ann_cnt += 1
        if not (float(anns[i][8]) >= 0.25):  # visibility.
            continue
        if not (int(anns[i][6]) == 1):  # whether ignore.
            continue
        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
            continue
        if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
            #category_id = -1
            continue

        category_id = 1  # pedestrian (non-static)
        if not track_id == tid_last:
            tid_curr += 1
            tid_last = track_id
        ann = {
            'id': ann_cnt,
            'category_id': category_id,
            'image_id': frame_id,
            'track_id': tid_curr,
            'bbox': anns[i][2:6].tolist(),
            'conf': float(anns[i][6]),
            'iscrowd': 0,
            'area': float(anns[i][4] * anns[i][5])
        }
        annotation_series.append(ann)
    return annotation_series


def _yxwh_to_xxyy(bbox):
    """
    yxwh: (y, x) of top left corner; width; height
    xxyy: xmin, xmax, ymin, ymax
    Please note: in image coordinates, x is vertical and y is horizontal.
    """
    tl_y, tl_x, bbox_w, bbox_h = bbox
    xmin = tl_x
    xmax = tl_x + bbox_h
    ymin = tl_y
    ymax = tl_y + bbox_w
    return (xmin, xmax, ymin, ymax)


def _xxyy_to_yxwh(xxyy):
    """
    xxyy: xmin, xmax, ymin, ymax
    yxwh: (y, x) of top left corner; width; height
    Please note: in image coordinates, x is vertical and y is horizontal.
    """
    xmin, xmax, ymin, ymax = xxyy
    tl_x = xmin
    bbox_h = xmax - xmin
    tl_y = ymin
    bbox_w = ymax - ymin

    return (tl_y, tl_x, bbox_w, bbox_h)


def _yxwh_to_xxyy_bounded(bbox, image_shape_xy):
    """
    `_yxwh_to_xxyy`, bounded by image shape
    """
    xmin, xmax, ymin, ymax = _yxwh_to_xxyy(bbox)
    image_x, image_y = image_shape_xy

    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, image_x), min(ymax, image_y)

    xmin, xmax, ymin, ymax = [int(num) for num in (xmin, xmax, ymin, ymax)]
    return (xmin, xmax, ymin, ymax)


def _iou(bbox1, bbox2):
    xmin1, xmax1, ymin1, ymax1 = _yxwh_to_xxyy(bbox1)
    xmin2, xmax2, ymin2, ymax2 = _yxwh_to_xxyy(bbox2)

    if xmin1 > xmax2 or xmin2 > xmax1 or ymin1 > ymax2 or ymin2 > ymax1:
        return 0

    xmin_intersc = max(xmin1, xmin2)
    xmax_intersc = min(xmax1, xmax2)
    ymin_intersc = max(ymin1, ymin2)
    ymax_intersc = min(ymax1, ymax2)

    area_bbox1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area_bbox2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    area_intersection = (xmax_intersc - xmin_intersc) * (ymax_intersc -
                                                         ymin_intersc)
    area_union = area_intersection + area_bbox1 + area_bbox2

    iou = area_intersection / area_union
    return iou


def find_isolated_tracklets(annotation_series,
                            first_k=10,
                            max_iou_allowed=0.1):
    isolated_tracklets = set([ann['track_id'] for ann in annotation_series])

    if first_k is not None:
        num_images = first_k
    else:
        num_images = len(
            np.unique([ann['image_id'] for ann in annotation_series]))

    for image_id in range(num_images):
        tracklet_list, bbox_list = [], []
        for ann in annotation_series:
            if ann['image_id'] != image_id:
                continue
            track_id = ann['track_id']
            assert track_id not in tracklet_list
            tracklet_list.append(track_id)
            bbox_list.append(ann['bbox'])
        for idx, bbox in enumerate(bbox_list):
            other_bboxes = bbox_list[:idx] + bbox_list[idx + 1:]
            for other_bbox in other_bboxes:
                if _iou(bbox, other_bbox) > max_iou_allowed:
                    isolated_tracklets.discard(
                        tracklet_list[idx])  # remove if exists

    return isolated_tracklets


def remove_bbox(image, bbox_to_remove):
    assert bbox_to_remove is not None
    xmin, xmax, ymin, ymax = \
        _yxwh_to_xxyy_bounded(bbox_to_remove, image.shape[:2])
    image[xmin:xmax, ymin:ymax, ...] = 0


def get_bbox(image, bbox):
    return _yxwh_to_xxyy_bounded(bbox, image.shape[:2])


def remove_mask(image, mask_to_remove):
    assert mask_to_remove is not None
    assert image.shape[0] == mask_to_remove.shape[0] and \
           image.shape[1] == mask_to_remove.shape[1]
    assert len(np.unique(mask_to_remove)) in [1, 2], \
        "Non-binary mask provided!"

    if mask_to_remove.max() == 255:
        mask_to_remove = mask_to_remove / 255

    image[mask_to_remove == 1, ...] = 0


def get_shifted_bbox(bbox, shift_xy, image_shape_xy):
    xmin, xmax, ymin, ymax = _yxwh_to_xxyy_bounded(bbox, image_shape_xy)
    xmin += shift_xy[0]
    xmax += shift_xy[0]
    ymin += shift_xy[1]
    ymax += shift_xy[1]

    image_x, image_y = image_shape_xy

    old_margins = (xmin, xmax, ymin, ymax)

    xmin, ymin = min(max(xmin, 0), image_x), min(max(ymin, 0), image_y)
    xmax, ymax = min(max(xmax, 0), image_x), min(max(ymax, 0), image_y)

    shifted_bbox = _xxyy_to_yxwh((xmin, xmax, ymin, ymax))
    margin_delta = [(xmin, xmax, ymin, ymax)[idx] - old_margins[idx]
                    for idx in range(4)]

    return shifted_bbox, margin_delta


def crop_patch_by_margin(image_patch, margin_delta):
    if margin_delta != [0, 0, 0, 0]:
        if margin_delta[0] < 0 or margin_delta[1] > 0 or margin_delta[
                2] < 0 or margin_delta[3] > 0:
            # patch completely outside image.
            return None

        if margin_delta[0] > 0:
            image_patch = image_patch[margin_delta[0]:, :, :]
        if margin_delta[1] < 0:
            image_patch = image_patch[:margin_delta[1], :, :]
        if margin_delta[2] > 0:
            image_patch = image_patch[:, margin_delta[2]:, :]
        if margin_delta[3] < 0:
            image_patch = image_patch[:, :margin_delta[3], :]

    return image_patch


def get_image_patch(image, bbox):
    xmin, xmax, ymin, ymax = _yxwh_to_xxyy_bounded(bbox, image.shape[:2])
    return image[xmin:xmax, ymin:ymax]


def paste_patch_by_bbox(background, patch, paste_loc_bbox):
    pasted = background.copy()
    if patch is not None:
        xmin, xmax, ymin, ymax = _yxwh_to_xxyy_bounded(paste_loc_bbox,
                                                       background.shape[:2])
        pasted[xmin:xmax, ymin:ymax, ...] = patch
    return pasted


def paste_masked_object(background, foreground, orig_mask, shift_xy):
    pasted = background.copy()

    assert background.shape[0] == foreground.shape[0] and \
           background.shape[1] == foreground.shape[1]
    assert background.shape[0] == orig_mask.shape[0] and \
           background.shape[1] == orig_mask.shape[1]
    assert len(np.unique(orig_mask)) in [1, 2], \
        "Non-binary mask provided!"
    if orig_mask.max() == 255:
        orig_mask = orig_mask / 255

    image_h, image_w = background.shape[:2]
    assert shift_xy[0] <= image_h and shift_xy[1] <= image_w

    # Pad the images and masks. This makes the masking easy.
    background = np.dstack([
        np.pad(background[:, :, c], ((image_h, image_h), (image_w, image_w)),
               mode='constant',
               constant_values=0) for c in range(background.shape[2])
    ])
    foreground = np.dstack([
        np.pad(foreground[:, :, c], ((image_h, image_h), (image_w, image_w)),
               mode='constant',
               constant_values=0) for c in range(foreground.shape[2])
    ])
    shifted_mask = np.pad(
        orig_mask, ((image_h + shift_xy[0] // 2, image_h - shift_xy[0] // 2),
                    (image_w + shift_xy[1] // 2, image_w - shift_xy[1] // 2)),
        mode='constant',
        constant_values=0)
    orig_mask = np.pad(orig_mask, ((image_h, image_h), (image_w, image_w)),
                       mode='constant',
                       constant_values=0)

    background[shifted_mask == 1, ...] = foreground[orig_mask == 1, ...]

    pasted = background[image_h:-image_h, image_w:-image_w, ...]
    shifted_mask = shifted_mask[image_h:-image_h, image_w:-image_w, ...]

    shifted_mask = shifted_mask * 255

    return pasted, shifted_mask


"""
TODO: Implement the following functions!!
"""


def dilate_bbox(bbox):
    return bbox


def shrink_bbox(bbox):
    return bbox


def dilate_mask(mask):
    return mask
