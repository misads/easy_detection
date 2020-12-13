from torch import optim
from easydict import EasyDict

schedulers = {
    'yolo2': {
        'base': 0.001 / 24,
        'epochs': [5, 60, 90, 99999999],  # 一共150个epoch
        'ratios': [0.24, 1, 0.1, 0.01]
    },
    'faster_rcnn': {
        'base': 0.001,
        'epochs': [1, 7, 10, 99999999],  # 一共12个epoch
        'ratios': [0.1, 1, 0.1, 0.01],
    },
}

schedulers['frcnn'] = schedulers['faster_rcnn']

schedulers = EasyDict(schedulers)

def get_scheduler(opt, optimizer):
    if opt.scheduler is None:
        opt.scheduler = opt.model.lower()

    if opt.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
    elif opt.scheduler.lower() == 'none':
        def lambda_decay(step) -> float:
            return 1.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)
    else:
        epochs = schedulers[opt.scheduler].epochs  # [1, 7, 10, 99999999]
        ratios = schedulers[opt.scheduler].ratios
        def lambda_decay(step) -> float:
            base = schedulers[opt.scheduler].base
            for epoch, ratio in zip(epochs, ratios):
                if step < epoch:
                    return base * ratio

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)        

    return scheduler
