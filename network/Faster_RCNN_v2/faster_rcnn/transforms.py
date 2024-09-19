import math

def size_divisible(size, divisor=32):
    """
    返回大于size且能被divisor整除的最小size
    """
    size = math.ceil(size / divisor) * divisor
    return int(size)


def batch_images(images, size_divisor=32):
    """
    将不同尺寸的图像组成相同尺寸的一个batch:
        先padding成图像中(最长宽x最长高)的尺寸, 检查尺寸是否为size_divisor的整数倍
    """

    image_sizes = zip(*[img.shape for img in images])
    max_sizes = [max(size) for size in image_sizes]

    max_sizes[1] = size_divisible(max_sizes[1], divisor=size_divisor)
    max_sizes[2] = size_divisible(max_sizes[2], divisor=size_divisor)

    for size in max_sizes[1:]:
        assert size % size_divisor == 0, f'Image must be padded to sizes which are divisible by {size_divisor}'

    batch_size = len(images)
    batch_shape = [batch_size] + max_sizes  # padding为最大尺寸的
    
    batched_imgs = images[0].new_zeros(batch_shape)
    for i in range(batch_size):
        img = images[i]
        batched_imgs[i][:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
    
    return batched_imgs