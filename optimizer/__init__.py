from .LookAhead import Lookahead
from .RAdam import RAdam
from .Ranger import Ranger
from torch import optim


def get_optimizer(config, module):
    optimizer = config.OPTIMIZE.OPTIMIZER
    lr = config.OPTIMIZE.BASE_LR

    if optimizer == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=lr, betas=(0.95, 0.999))
    elif optimizer == 'sgd':  # 从头训练 lr=0.1 fine_tune lr=0.01
        # optimizer = optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005*24)  # Yolo
        optimizer = optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)  # FRCNN
    elif optimizer == 'radam':
        optimizer = RAdam(module.parameters(), lr=lr, betas=(0.95, 0.999))
    elif optimizer == 'lookahead':
        optimizer = Lookahead(module.parameters())
    elif optimizer == 'ranger':
        optimizer = Ranger(module.parameters(), lr=lr)
    else:
        raise NotImplementedError

    return optimizer