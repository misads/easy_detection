import os
from abc import abstractmethod

import torch
import warnings
import sys

from misc_utils import color_print, progress_bar
from options import opt
from utils import deprecated
from mscv.image import tensor2im
from mscv.aug_test import tta_inference, tta_inference_x8


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def inference(self, x, progress_idx=None):
        # x: Tensor([1, C, H, W])
        # recovered: 直接出图，可以用Image.save保存
        with torch.no_grad():
            img_var = x.to(device=opt.device)

            if opt.tta:
                output = tta_inference(self.forward, img_var, 10, 10, 256, 256,
                                       progress_idx=progress_idx).unsqueeze(0)
                recovered = tensor2im(output)
            elif opt.tta_x8:
                output = tta_inference_x8(self.forward, img_var, 10, 10, 256, 256,
                                          progress_idx=progress_idx).unsqueeze(0)
                recovered = tensor2im(output)
            else:
                recovered = self.forward(img_var)
                if isinstance(recovered, tuple) or isinstance(recovered, list):
                    recovered = recovered[0]

                recovered = tensor2im(recovered)

        return recovered

    @abstractmethod
    def load(self, ckpt_path):
        pass

    @abstractmethod
    def save(self, which_epoch):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    # helper saving function that can be used by subclasses
    @deprecated('model.save_network() is deprecated now, use model.save() instead')
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    @deprecated('model.load_network() is deprecated now, use model.load() instead')
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            color_print("Exception: Checkpoint '%s' not found" % save_path, 1)
            if network_label == 'G':
                raise Exception("Generator must exist!,file '%s' not found" % save_path)
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path, map_location=opt.device))
                color_print('Load checkpoint from %s.' % save_path, 3)
            
            except:
                pretrained_dict = torch.load(save_path, map_location=opt.device)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

