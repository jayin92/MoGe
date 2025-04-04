import sys
sys.path.append("submodules/MoGe")

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np
import torch
from torchvision import transforms as tvt
from PIL.Image import Image as PILImage
from typing import List
from tqdm import tqdm

# make moge is module

from moge.model import MoGeModel

class MoGeIDU:
    def __init__(self, save_path, device, fov_x=60.0):
        self.save_path = save_path
        self.device = device
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
        self.model.output_mask = True
        self.fov_x = fov_x
        os.makedirs(save_path, exist_ok=True)
    
    def __del__(self):
        if self.model is not None:
            try:
                self.model.to("cpu")  # Move to CPU before deleting
                del self.model
            except Exception as e:
                print(f"Error during model cleanup: {e}")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, refined_imgs: List[PILImage], pbar=True) -> List[np.ndarray]:
        depths = []
        for idx, img in enumerate(tqdm(refined_imgs, desc=f"Generate depth maps to {self.save_path}", disable=not pbar)):
            img_tensor = tvt.ToTensor()(img).to(self.device)
            output = self.model.infer(img_tensor, fov_x=self.fov_x)
            assert 'mask' in output, "Model output does not contain mask"
            depth = output['depth'].cpu().numpy()
            cv2.imwrite(os.path.join(self.save_path, '{0:05d}'.format(idx) + ".exr"), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            depths.append(depth)
            del output
        
        return depths
