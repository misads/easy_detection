from collections import OrderedDict
import torch.nn.functional as F
from torch import nn

class FPN(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), 
                       out_channels=256, 
                       extra_blocks='pool'):
        """
        Args:
            in_channels: list(int)
            out_channels: int
            extra_blocks: str 'pool' or None
        """
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        for in_channel in in_channels:
            self.inner_blocks.append(nn.Conv2d(in_channel, out_channels, kernel_size=1))

        self.layer_blocks = nn.ModuleList()
        for in_channel in in_channels:
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.extra_blocks = extra_blocks

    def forward(self, x):
        """
        FPN示意图:
                                                                     'pool'
                           (1x1)    <1/32>                (3x3)         ↑
            layer4  ━━> 2048 -> 256  ━━>  feat_4  ━━>  256 -> 256  ━━>  3
              △                             ↓
              ┃                          upsample
              ┃            (1x1)    <1/16>  ↓             (3x3)
            layer3  ━━> 1024 -> 256  ━━━━━> + ━━━━━━>  256 -> 256  ━━>  2
              △                             ↓
              ┃                          upsample
              ┃            (1x1)    <1/8>   ↓             (3x3)
            layer2  ━━>  512 -> 256  ━━━━━> + ━━━━━━>  256 -> 256  ━━>  1
              △                             ↓
              ┃                          upsample
              ┃            (1x1)    <1/4>   ↓             (3x3)
            layer1  ━━>  256 -> 256  ━━━━━> + ━━━━━━>  256 -> 256  ━━>  0
              △ 
              ┃
              ┃
            image

        outputs: [0], [1], [2], [3], ['pool']
        
        Args:
            x: OrderedDict
        """
        outputs = OrderedDict()
        out_layers = []
        top_down = None
        for i in range(len(x)-1, -1, -1):
            layer = x[i]
            feature = self.inner_blocks[i](layer)
            if i != len(x) - 1 :
                feature = feature + top_down

            if i != 0:
                top_down = self.upsample(feature)

            out_layers.insert(0, self.layer_blocks[i](feature))   

        for i in range(len(out_layers)):
            outputs[i] = out_layers[i]
            
        if self.extra_blocks == 'pool':
            outputs['pool'] = F.max_pool2d(out_layers[-1], 1, 2, 0)

        return outputs
