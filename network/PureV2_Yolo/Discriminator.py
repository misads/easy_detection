import torch
import torch.nn as nn


class D(nn.Module):
    def __init__(self, nc=6, nf=64):
        super(D, self).__init__()

        main = nn.Sequential()
        # 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

        # 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        main.add_module(name, blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module(name, blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, nf * 2, 4, 1, 1, bias=False))
        main.add_module('%s_bn' % name, nn.BatchNorm2d(nf * 2))

        # 31
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
        main.add_module('%s_sigmoid' % name, nn.Sigmoid())
        # 30 (sizePatchGAN=30)

        self.main = main

    def forward(self, x):
        output = self.main(x)
        return output


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block
