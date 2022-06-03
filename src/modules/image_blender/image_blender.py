import os

import cv2
import numpy as np
import torch
from src.image_utils import get_image_patch, paste_patch_by_bbox

from .iharm.inference.predictor import Predictor
from .iharm.inference.utils import load_model


class ArgParseReplacer(object):

    def __init__(self):
        self.model_type = 'improved_ssam256'
        self.checkpoint = os.path.join(
            os.path.dirname(__file__),
            'checkpoints/fixed256/improved_ssam256/issam256.pth')
        self.resize = 256


class ImageBlender(object):

    def __init__(self):
        args = ArgParseReplacer()

        self.resize_shape = (args.resize, ) * 2

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        net = load_model(args.model_type, args.checkpoint)
        self.image_blender = Predictor(net, self.device)

    def blend_full_image(self, image, mask):
        """
        The image will not be resized in this function.
        """
        mask[mask <= 100] = 0
        mask[mask > 100] = 1
        mask = mask.astype(np.float32)

        blended_image = self.image_blender.predict(image, mask)

        return blended_image

    def blend_image(self, image, mask, bbox_fov):
        """
        The image will be resized prior to inference and resized back.
        bbox_fov: The field-of-view for image blending.
        """

        # bbox format: (y, x) of top left point, width, height.
        _, _, bbox_fov_w, bbox_fov_h = bbox_fov

        if bbox_fov_w <= 0 or bbox_fov_h <= 0:
            return image

        image_patch = get_image_patch(image, bbox_fov)
        mask_patch = get_image_patch(mask, bbox_fov)
        if image_patch.shape[0] == 0 or image_patch.shape[1] == 0:
            return image

        patch_h, patch_w = image_patch.shape[:2]
        if self.resize_shape[0] > 0:
            image_patch = cv2.resize(image_patch, self.resize_shape,
                                     cv2.INTER_LINEAR)
        if self.resize_shape[0] > 0:
            mask_patch = cv2.resize(mask_patch, self.resize_shape,
                                    cv2.INTER_LINEAR)
        mask_patch[mask_patch <= 100] = 0
        mask_patch[mask_patch > 100] = 1
        mask_patch = mask_patch.astype(np.float32)

        blended_image_patch = self.image_blender.predict(
            image_patch, mask_patch)

        # Resize prediction back to original size.
        blended_image_patch = cv2.resize(blended_image_patch,
                                         (patch_w, patch_h), cv2.INTER_LINEAR)

        image = paste_patch_by_bbox(background=image,
                                    patch=blended_image_patch,
                                    paste_loc_bbox=bbox_fov)

        return image
