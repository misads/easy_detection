import torch
from torch import nn

class AnchorGenerator(nn.Module):
    def __init__(
        self, scales=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()

        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    @staticmethod
    def generate_base_anchors(scale, aspect_ratios, device="cpu"):
        """
        说明:
            对单个scale和aspect_ratios生成基本anchor, 基本anchor以原点为中心

        Args:
            scale: int
            aspect_ratios: list(float)
            device: str or torch.device

        Returns: 
            [[-x, -ax, x, ax], ...]  # base anchor以原点(0, 0)为中心

        Example:
            generate_base_anchors(32, [0.5, 1.0, 2.0]) = 
            tensor([[-23., -11.,  23.,  11.],
                    [-16., -16.,  16.,  16.],
                    [-11., -23.,  11.,  23.]])
        """
        assert isinstance(aspect_ratios, list) or isinstance(aspect_ratios, tuple)

        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32, device=device)

        """
        假设宽高比 aspect_ratio = a
        假设 scale = S, anchor面积为 S^2

            (anchor示意图)
                ┏━━╋━━┓
                ┃  ┃  ┃ ax
             ━━━╋━━╋━━╋━━━ 
                ┃  ┃  ┃ ax
                ┗━━╋━━┛
                 x ┃ x
           
            2x · 2ax = S^2
            4a · x^2 = S^2
            x = S / (2√a)
            ax = √α · S/2
        """

        sqrt_a = torch.sqrt(aspect_ratios)
        ws = scale / (2 * sqrt_a)  # x
        hs = sqrt_a * scale / 2  # ax

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)   
        return base_anchors.round()

    def set_cell_anchors(self, device='cpu'):
        """
        生成len(sizes) * len(aspect_ratios)个 cell_anchors (均以原点为中心)
        """
        if self.cell_anchors is not None:
            return self.cell_anchors

        cell_anchors = [
            self.generate_base_anchors(scale, self.aspect_ratios, device) for scale in self.scales
        ]
        self.cell_anchors = cell_anchors  # 个tensor

    # def num_anchors_per_location(self):
    #     return [len(self.aspect_ratios) for _ in self.scales]

    def grid_anchors(self, image_size, feature_maps):
        """
        说明:
            feature maps:
                        feat_width
                ┏━━━┳━━━┳━━━┳━━━┳━━━┳━━━┓  ┏━━━┳━━━┳━━━┳━━━┳━━━┓  ┏━━━┳━━━┳━━━┓
                ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ * ┃ * ┃ * ┃
                ┣━━━╋━━━╋━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━┫
                ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ * ┃ * ┃ * ┃
    feat_height ┣━━━╋━━━╋━━━╋━━━╋━━━╋━━━┫  ┣━━━╋━━━╋━━━╋━━━╋━━━┫  ┗━━━┻━━━┻━━━┛
                ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ * ┃ * ┃ * ┃ * ┃ * ┃  
                ┣━━━╋━━━╋━━━╋━━━╋━━━╋━━━┫  ┗━━━┻━━━┻━━━┻━━━┻━━━┛  
                ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃
                ┗━━━┻━━━┻━━━┻━━━┻━━━┻━━━┛

            在feature maps中, 每个像素中心(图中*处)可以作为anchor的中心。(即 feat_width = num_anchor_x)
            stride_width, stride_height 是每个anchor的步长, 总共(feat_width * feat_height)个anchors要覆盖整幅图像。
            
                num_anchor_x * stride_width = image_width
                num_anchor_y * stride_height = image_height
                
        Args:
            image_size: tuple(int)  height, width
            feature_maps: list(Tensor)

        Returns:
            anchors: list(Tensor)

        """
        anchors = []
        assert len(feature_maps) == len(self.cell_anchors), ("There needs to be a match between "
            f"the number of feature maps passed (len={len(feature_maps)}) and the number of anchor "
            f"scales specified ({str(self.scales)} len={len(self.cell_anchors)}).")

        for i in range(len(feature_maps)):
            image_height, image_width = image_size
            feature_map = feature_maps[i]

            # 见上图及说明
            # feat_width = num_anchor_x
            # feat_height = num_anchor_y
            num_anchor_y, num_anchor_x = feature_map.shape[-2:]  

            stride_height = image_height / num_anchor_y
            stride_width = image_width / num_anchor_x
            
            base_anchors = self.cell_anchors[i]  # 只有一种尺寸, 多个scale
            device = base_anchors.device

            shifts_y = torch.arange(0, num_anchor_y, dtype=torch.int32, device=device) * stride_height
            shifts_x = torch.arange(0, num_anchor_x, dtype=torch.int32, device=device) * stride_width

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                # shifts在前, ratios在后  [num_shifts, num_ratios, 4]
                # 对应 features 中 W, H 在前, C 在后
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

        
    def forward(self, images, feature_maps):
        """
        Args:
            images: Tensor [b, c, h, w] shape
            feature_maps: list(Tensor)
        """
        image_size = images.shape[-2:]
        
        self.set_cell_anchors(feature_maps[0].device)
        anchors_over_all_feature_maps = self.grid_anchors(image_size, feature_maps)  # list(Tensor)

        anchors_concated = torch.cat(anchors_over_all_feature_maps)  # Tensorf of [N, 4] 

        anchors = []
        for _ in range(len(images)):
            anchors.append(anchors_concated)

        return anchors