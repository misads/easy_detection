import os
from abc import abstractmethod

import torch

from misc_utils import color_print
from options import opt
from options.helper import is_distributed, is_first_gpu
from optimizer import get_optimizer
from scheduler import get_scheduler

from mscv import ExponentialMovingAverage, load_checkpoint, save_checkpoint, print_network


class BaseModel(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, sample, *args):
        if self.training:
            return self.update(sample, *args)
        else:
            return self.forward_test(sample, *args)

    @property
    def detector(self):
        return self._detector

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    def init_common(self):
        if opt.debug and is_first_gpu():
            print_network(self.detector)

        self.to(opt.device)
        # 多GPU支持
        if is_distributed():
            self._detector = torch.nn.parallel.DistributedDataParallel(self._detector, find_unused_parameters=False,
                    device_ids=[opt.local_rank], output_device=opt.local_rank)
            # self.detector = torch.nn.parallel.DistributedDataParallel(self.detector, device_ids=[opt.local_rank], output_device=opt.local_rank)

        self._optimizer = get_optimizer(self.detector, self.config)
        self._scheduler = get_scheduler(self.optimizer, self.config)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join('checkpoints', opt.tag)


    @abstractmethod
    def update(self, sample: dict, *args, **kwargs):
        """
        这个函数会计算loss并且通过optimizer.step()更新网络权重。
        """
        pass

    @abstractmethod
    def forward_test(self, sample, *args):
        """
        这个函数会由输入图像给出一个batch的预测结果。

        Args:
            sample: dict
                ['ori_image'] (list)
                ['ori_sizes'] (list)
                ['image'] (list)
                ['bboxes'] (list)
                ['labels'] (list)
                ['path'] (list)
                ['yolo_boxes'] (Tensor) shape=(N, 250)
                ['yolo4_boxes'] (Tensor) shape=(N, 50, 5)
                ['yolo5_boxes'] (Tensor) shape=(N, 6)

        Returns:
            tuple: (batch_bboxes, batch_labels, batch_scores)
            
            batch_bboxes: list(np.ndarray), (N, [-1, 4])
                一个batch的预测框, xyxy格式

            batch_labels: list(np.ndarray), (N, [-1])
                一个batch的预测标签, np.int32格式

            batch_scores: list(np.ndarray), (N, [-1])
                一个batch的预测分数, np.float格式
        """
        pass

    def load(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.detector.load_state_dict(state_dict['detector'])
        color_print('Load checkpoint from %s.' % ckpt_path, 3)

        if opt.resume or 'RESUME' in self.config.MISC:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

            color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)

        return 0

        if ckpt_path[-2:] != 'pt':
            return 0
            
        load_dict = {
            'detector': self.detector,
        }

        if opt.resume or 'RESUME' in self.config.MISC:
            load_dict.update({
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
            })
            color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            color_print('Load checkpoint from %s.' % ckpt_path, 3)

        ckpt_info = load_checkpoint(load_dict, ckpt_path, map_location='cpu')

        if opt.resume or 'RESUME' in self.config.MISC:
            self.optimizer.load_state_dict(load_dict['optimizer'])
            self.scheduler.step()

        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch, published=False):
        save_filename = f'{which_epoch}_{self.config.MODEL.NAME}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'detector': self.detector,
            'epoch': which_epoch
        }
        
        if published:
            save_dict['epoch'] = 0
        else:
            save_dict['optimizer'] = self.optimizer
            save_dict['scheduler'] = self.scheduler

        save_checkpoint(save_dict, save_path)
        color_print(f'Save checkpoint "{save_path}".', 3)

