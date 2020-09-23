from torch import optim


def get_scheduler(opt, optimizer):
    if opt.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
    elif opt.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochs // 3, gamma=0.1)
    elif opt.scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif opt.scheduler == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opt.lr, max_lr=0.1 * opt.lr,
                                                step_size_up=opt.epochs // 10,
                                                step_size_down=opt.epochs // 10)
    elif opt.scheduler == 'lambda':
        def lambda_decay(step) -> float:
            warm_steps = 500
            if step <= warm_steps:
                return step / warm_steps
            else:
                return 0.997 ** ((step - warm_steps) // 1000)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)

    else:
        scheduler = None

    return scheduler
