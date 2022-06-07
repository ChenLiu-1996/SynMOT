import logging
import os
import sys
import uuid

from src.modules.image_inpainter.lama.saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from src.image_utils import get_bbox
from src.modules.image_inpainter.lama.saicinpainting.training.trainers import load_checkpoint

logger = logging.getLogger(__name__)


class ImageInpainter(object):

    def __init__(self):
        # Establish relevant paths
        checkpoint_dir = os.path.join(
            os.path.dirname(__file__),
            'checkpoints/image_inpainter/')
        config_path = os.path.join(
            os.path.dirname(__file__),
            'lama/configs/default_prediction.yaml')
        with open(config_path, 'r') as f:
            self.config = OmegaConf.create(yaml.safe_load(f))
        if not os.path.exists(checkpoint_dir):
            logger.error("ERROR: Pretrained model does not exist under SynMOT/src/modules/image_inpainter/checkpoints/, follow README for instructions")
            sys.exit(1)

        # Load the model once into memory
        self.device = torch.device(self.config.device)

        model_config_path = os.path.join(checkpoint_dir, 'config.yaml')
        with open(model_config_path, 'r') as f:
            model_config = OmegaConf.create(yaml.safe_load(f))
        model_config.training_model.predict_only = True
        model_config.visualizer.kind = 'noop'
        out_ext = self.config.get('out_ext', '.png')

        model_path = os.path.join(checkpoint_dir, 'models/best.ckpt')
        self.model = load_checkpoint(model_config, model_path, strict=False, map_location='cpu')

        self.model.freeze()
        self.model.to(self.device)


    def reshape_image(self, image, return_orig=False):
        image_dims = image.shape
        if len(image_dims) == 2:
            image = image.reshape(image_dims[0], image_dims[1], 1)
        image = np.transpose(image, (2, 0, 1))
        out_img = image.astype('float32') / 255
        if return_orig:
            return out_img, img
        return  out_img


    def generate_masked_img(self, ref_img, mask):
        mask_img = np.zeros(ref_img.shape)
        xmin, xmax, ymin, ymax = get_bbox(ref_img, mask)
        mask_img[xmin:xmax, ymin:ymax, ...] = 255
        return mask_img[:,:,0]


    def inpaint_image(self, image, mask):
        cur_res = image
        #mask = self.generate_masked_img(image, mask)
        image = self.reshape_image(image)
        mask = self.reshape_image(mask)
        try:
            with torch.no_grad():
                sample = {'image': image, 'mask': mask}
                batch = move_to_device(default_collate([sample]), self.device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = self.model(batch)
                cur_res = batch[self.config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                # Convert image back to CV2 compatible format
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                #cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                return cur_res
        except Exception as ex:
            logging.exception('Inpainting failed')
        return cur_res

