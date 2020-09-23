import torch.nn as nn
from loss.gradient import grad_loss
from loss.vggloss import vgg_loss
from loss.ssim import ssim_loss as criterionSSIM
from options import opt

criterionCAE = nn.L1Loss()
criterionL1 = criterionCAE
criterionBCE = nn.BCELoss()
criterionMSE = nn.MSELoss()


def get_default_loss(recovered, y, avg_meters):
    ssim = - criterionSSIM(recovered, y)
    ssim_loss = ssim * opt.weight_ssim

    # Compute L1 loss (not used)
    l1_loss = criterionL1(recovered, y)
    l1_loss = l1_loss * opt.weight_l1

    loss = ssim_loss + l1_loss

    # record losses
    avg_meters.update({'ssim': -ssim.item(), 'L1': l1_loss.item()})

    if opt.weight_grad:
        loss_grad = grad_loss(recovered, y) * opt.weight_grad
        loss += loss_grad
        avg_meters.update({'gradient': loss_grad.item()})

    if opt.weight_vgg:
        content_loss = vgg_loss(recovered, y) * opt.weight_vgg
        loss += content_loss
        avg_meters.update({'vgg': content_loss.item()})

    return loss