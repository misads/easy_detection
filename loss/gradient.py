import torch
import torch.nn as nn

criterionCAE = nn.L1Loss()


def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y


def grad_loss(recovered, label):
    gradie_h_est, gradie_v_est = gradient(recovered)
    gradie_h_gt, gradie_v_gt = gradient(label)

    L_tran_h = criterionCAE(gradie_h_est, gradie_h_gt)
    L_tran_v = criterionCAE(gradie_v_est, gradie_v_gt)

    return (L_tran_h + L_tran_v)