from .LookAhead import Lookahead
from .RAdam import RAdam
from .Ranger import Ranger
from torch import optim


def get_optimizer(opt, module):
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=opt.lr, betas=(0.95, 0.999))
    elif opt.optimizer == 'sgd':  # 从头训练 lr=0.1 fine_tune lr=0.01
        optimizer = optim.SGD(module.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    elif opt.optimizer == 'radam':
        optimizer = RAdam(module.parameters(), lr=opt.lr, betas=(0.95, 0.999))
    elif opt.optimizer == 'lookahead':
        optimizer = Lookahead(module.parameters())
    elif opt.optimizer == 'ranger':
        optimizer = Ranger(module.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError

    return optimizer