from itertools import product

import torch
from math import sqrt


class Config512:
    ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    BOXES_PER_LOCATION = [4, 6, 6, 6, 6, 4, 4]
    CLIP = True
    FEATURE_MAPS = [64, 32, 16, 8, 4, 2, 1]
    MAX_SIZES = [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.65]
    MIN_SIZES = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
    STRIDES = [8, 16, 32, 64, 128, 256, 512]


class Config300:
    ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]
    CLIP = True
    FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
    MAX_SIZES = [60, 111, 162, 213, 264, 315]
    MIN_SIZES = [30, 60, 111, 162, 213, 264]
    STRIDES = [8, 16, 32, 64, 100, 300]


class PriorBox:
    def __init__(self, config):
        self.image_size = config.DATA.SCALE
        if config.DATA.SCALE == 300:
            prior_config = Config300()
        elif config.DATA.SCALE== 512:
            prior_config = Config512()
        else:
            raise RuntimeError

        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
