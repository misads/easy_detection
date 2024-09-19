import torch

class BalancedSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched_idxs: list(Tensor) (N, [len(anchors_per_image)]) 取值为1, 0或-1
                 1:     正样本(iou>0.7(如果没有, 则取iou最大的))
                 0:     背景(负样本, iou<0.3)
                -1:     丢弃(0.3<=iou<0.7)

        Returns:
            pos_masks list(tensor)  (N, [len(anchors_per_image)])
            neg_masks list(tensor)  (N, [len(anchors_per_image)])

            返回仅包含0, 1的mask, 长度与len(anchors_per_image)相同, 
            0表示该下标不是正(负)样本, 1表示该下标为正(负)样本
        """
        pos_masks = []
        neg_masks = []
        for matched_idxs_per_image in matched_idxs:

            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_positive = int(self.batch_size_per_image * self.positive_fraction)
            # 如果正样本不够, 则所有的正样本都用上
            num_positive = min(positive.numel(), num_positive)

            num_negative = self.batch_size_per_image - num_positive
            num_negative = min(negative.numel(), num_negative)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_positive]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_negative]

            sampled_pos_idx = positive[perm1]
            sampled_neg_idx = negative[perm2]

            # create binary mask from indices
            pos_label = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_label = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_label[sampled_pos_idx] = 1
            neg_label[sampled_neg_idx] = 1

            pos_masks.append(pos_label)
            neg_masks.append(neg_label)

        return pos_masks, neg_masks

    def __repr__(self):
        return (f'{self.__class__.__name__}'
            f'(batch_size_per_image={self.batch_size_per_image}, '
            f'positive_fraction={self.positive_fraction})')