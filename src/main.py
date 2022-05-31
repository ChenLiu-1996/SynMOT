import os
import random
import sys
from typing import Dict, Iterable, List

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/modules/human_segmenter/')
sys.path.append('../src/modules/image_inpainter/')
sys.path.append('../src/modules/image_blender/')
sys.path.append('../src/modules/human_segmenter/checkpoints/')
from src.image_utils import crop_patch_by_margin, find_isolated_tracklets, get_image_patch, get_shifted_bbox, \
    load_annotation_series, load_image_series, paste_image_patch, remove_bbox
from src.modules.human_segmenter import HumanSegmenter

MOT_DATA_FOLDER = '../datasets/mot/train/'
SEQ_FOLDER = 'MOT17-11-DPM'
OUTPUT_BLENDING_MASK_FOLDER = '../output/blending_masks/'
OUTPUT_IMAGE_FOLDER = '../output/images/'
OUTPUT_VIDEO_FOLDER = '../output/video/'


def shift_trajectories(image_series: List[np.array],
                       annotation_series: List[Dict],
                       moving_objects_ids: Iterable[int],
                       video_writer: cv2.VideoWriter):
    """
    Please note:
        This is incomplete at the moment.
        The code might not be performing the desired actions.

    Desired actions:
    For each human object selected as the `moving_objects`, we will
        1. Find its original `bbox` from the video.
        2. Perform human segmentation within `bbox` -> get `mask`.
           Detail: in `human_segmenter.process_image`,
                   perhaps set `bbox_outer` as dilated `bbox` and
                   `bbox_guess` as shrinked `bbox` for better segmentation.
        3. Remove the human object from its `mask`.
           Detail: perhaps dilate `mask` for cleaner removal.
        4. Fill the missing vacancy of the removed human object using image inpainting.
        5. Calculate the human object's destintation after shifting.
           Shifting amount is a constant, pre-defined value for each object.
           Detail: need to handle edge cases such as out-of-boundary issues.
        6. Paste the segmented human object to the destination.
        7. Harmonize the pasted human object and its surroundings using image blending.
        8. Update the annotation series to reflect the updated bbox location.
    """

    # Human segmentation model
    human_segmenter = HumanSegmenter()
    # Image inpainting model
    image_inpainter = None  # ImageInpainter()
    # Image blending model
    image_blender = None  # ImageBlender()

    # Pre-define how much to shift each object.
    image_h, image_w, _ = image_series[0].shape
    max_delta_h, max_delta_w = image_h / 2, image_w / 2
    # Hashmap: object id -> (x, y) shift for that object.
    shift_xy_by_object = dict([(k, (
        int(random.uniform(-max_delta_h, max_delta_h)),
        int(random.uniform(-max_delta_w, max_delta_w)),
    )) for k in sorted(moving_objects_ids)])

    for image_idx in tqdm(range(len(image_series))):
        adjusted_image = image_series[image_idx].copy()
        blending_mask = np.zeros_like(image_series[image_idx][:, :, 0])

        # Step 1. Remove the original bbox of all moving objects.
        for ann in annotation_series:
            if ann['image_id'] - 1 != image_idx:
                continue
            if not ann['track_id'] in moving_objects_ids:
                continue
            remove_bbox(adjusted_image, bbox_to_remove=ann['bbox'])

        # Step 2. Copy-Paste for moving objects.
        for ann in annotation_series:
            if ann['image_id'] - 1 != image_idx:
                continue
            if not ann['track_id'] in moving_objects_ids:
                continue

            image_patch_loc_bbox = ann['bbox']
            image_patch = get_image_patch(image_series[image_idx],
                                          image_patch_loc_bbox)

            # Note: after shifting, the bbox may get outside the boundaries.
            # Hence we record `margin_delta` to help crop the image patch accordingly.
            image_paste_loc_bbox, margin_delta = get_shifted_bbox(
                image_patch_loc_bbox, shift_xy_by_object[ann['track_id']],
                image_series[0].shape[:2])

            image_patch = crop_patch_by_margin(image_patch, margin_delta)

            paste_image_patch(adjusted_image,
                              image_patch=image_patch,
                              paste_loc_bbox=image_paste_loc_bbox)

            blending_mask = human_segmenter.process_image(
                image=adjusted_image,
                mask=blending_mask,
                bbox_outer=image_paste_loc_bbox,
                bbox_guess=image_paste_loc_bbox)

        # Save image and mask
        cv2.imwrite(
            OUTPUT_IMAGE_FOLDER + "image_" + str(image_idx).zfill(5) + ".png",
            adjusted_image)
        cv2.imwrite(
            OUTPUT_BLENDING_MASK_FOLDER + "mask_" + str(image_idx).zfill(5) +
            ".png", blending_mask)

        # Collect image into video stream
        if video_writer is not None:
            video_writer.write(adjusted_image)

    return video_writer


if __name__ == '__main__':
    random.seed(0)

    for folder in [
            OUTPUT_IMAGE_FOLDER, OUTPUT_BLENDING_MASK_FOLDER,
            OUTPUT_VIDEO_FOLDER
    ]:
        os.makedirs(folder, exist_ok=True)

    num_frames = None
    image_series = load_image_series(MOT_DATA_FOLDER,
                                     seq=SEQ_FOLDER,
                                     first_k=num_frames)
    annotation_series = load_annotation_series(MOT_DATA_FOLDER,
                                               seq=SEQ_FOLDER,
                                               first_k=num_frames)

    isolated_tracklets = find_isolated_tracklets(annotation_series,
                                                 first_k=num_frames)

    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FOLDER +
                                   '%s_shift_trajectories.mp4' % SEQ_FOLDER,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps=30,
                                   frameSize=image_series[0].shape[:2][::-1])

    video_writer = shift_trajectories(
        image_series=image_series,
        annotation_series=annotation_series,
        moving_objects_ids=isolated_tracklets,
        video_writer=video_writer,
    )

    video_writer.release()
