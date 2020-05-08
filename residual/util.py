import torch


def gen_mask(shape, input_size=128, output_size=192):
    """
    Generate mask for given input and output size. Assumes square images.
    Uses expand_size, which is distance from provided center of image to end of hallucination in pixels.

    :param shape: Shape of mask tensor to be generated like (num, channel, height, width)
    :param input_size:
    :param output_size:
    :returns: Mask tensor where pixels indicating mask are filled with 1. and rest are 0.
    """
    expand_size = (output_size - input_size) // 2
    mask = torch.zeros(shape)
    for i in range(shape[0]):
        mask[i, :, expand_size:, :] = 1.
        mask[i, :, :, :expand_size] = 1.
        mask[i, :, -expand_size:, :] = 1.
        mask[i, :, :, -expand_size:] = 1.
    return mask
