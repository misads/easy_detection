from collections import OrderedDict
import torch.nn.functional as F
from torch import nn

class FPN(nn.Module):
    def __init__(self, C3_size=512, 
                       C4_size=1024, 
                       C5_size=2048, 
                       feature_size=256):
        """
        Args:
            in_channels: list(int)
            out_channels: int
        """
        super(FPN, self).__init__()
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        FPN示意图:                                    (3x3, stride=2)
                                        ┏━━━> ReLU ━━━> 256 -> 256 ━━>  4
                      (3x3, stride=2)   ┃
              ┏━━━━━━━> 2048 -> 256  ━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━>  3  
              ┃                                                  
              ┃            (1x1)    <1/32>                (3x3)      
             C5 ━━━━━━> 2048 -> 256  ━━━>  P5_x  ━━━>  256 -> 256  ━━>  2
              △                             ↓
              ┃                          upsample
              ┃            (1x1)    <1/16>  ↓             (3x3)
             C4 ━━━━━━> 1024 -> 256  ━━━━━> + ━━━━━━>  256 -> 256  ━━>  1
              △                             ↓
              ┃                          upsample
              ┃            (1x1)    <1/8>   ↓             (3x3)
             C3 ━━━━━━>  512 -> 256  ━━━━━> + ━━━━━━>  256 -> 256  ━━>  0
              △                            
              ┃                         
              ┃            
             C2 (layer1)
              △ 
              ┃
              ┃
            image

        outputs: [0], [1], [2], [3], [4]
        
        Args:
            x: OrderedDict
        """
        outputs = OrderedDict()
        for i in range(5):
            outputs[i] = {}

        _, C3, C4, C5 = x[0], x[1], x[2], x[3]

        P5_x = self.P5_1(C5)
        P5_top_down = self.upsample(P5_x)
        outputs[2] = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_top_down + P4_x
        P4_top_down = self.upsample(P4_x)
        outputs[1] = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_top_down
        outputs[0] = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        outputs[3] = P6_x

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        outputs[4] = P7_x

        return outputs
