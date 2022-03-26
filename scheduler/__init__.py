from torch import optim
from easydict import EasyDict
from options import opt

schedulers = {
    '1x': {  # Faster_RCNN
        'epochs': [8, 11, 13],  # 12个epoch
        'ratios': [1, 0.1, 0.01],
    },
    '2x': {
        'epochs': [16, 22, 24],  # 24个epoch
        'ratios': [1, 0.1, 0.01],
    },
    '5x': { 
        'epochs': [40, 55, 60],  # 60个epoch
        'ratios': [1, 0.1, 0.01],
    },
    '10x': {  
        'epochs': [80, 110, 130],  # 120个epoch
        'ratios': [1, 0.1, 0.01],
    },
    '20e': {
        'epochs': [16, 19, 20],  # 20个epoch
        'ratios': [1, 0.1, 0.01],
    },
    '100e': {  # Yolo2和Yolo3
        'epochs': [80, 95, 100],  # 100个epoch
        'ratios': [1, 0.1, 0.01],
    },
    '1e': { # debug
        'epochs': [1],
        'ratios': [1],
    }

}

schedulers = EasyDict(schedulers)

def get_scheduler(config, optimizer):
    scheduler = config.OPTIMIZE.SCHEDULER

    if scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
    elif scheduler.lower() == 'none':
        def lambda_decay(step) -> float:
            return 1.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)
    else:
        if scheduler not in schedulers:
            if scheduler[-1] == 'x': 
                times = int(scheduler[:-1])
                schedulers[scheduler] = {
                    'epochs': [s * times for s in schedulers['1x'].epochs],
                    'ratios': [0.1, 1, 0.1, 0.01],
                }
                schedulers[scheduler]['epochs'][0] = 1

        epochs = schedulers[scheduler].epochs  # [1, 7, 10, 99999999]
        opt.epochs = epochs[-1]
        ratios = schedulers[scheduler].ratios
        
        def lambda_decay(step) -> float:
            for epoch, ratio in zip(epochs, ratios):
                if step < epoch:
                    return ratio
            return ratios[-1]

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)        

    return scheduler
