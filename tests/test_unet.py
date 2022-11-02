import unittest

# import matplotlib.pyplot as plt
import torch

from FrEIA.iunets import iUNet, IUnetCOnvBlock


def checkerboard(shape, k):
    """
        shape: dimensions of output tensor
        k: edge size of square
    """

    def indices(h, w):
        return torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)))

    h, w = shape
    k = int(k)
    base = indices(h // k, w // k).sum(dim=0) % 2
    x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
    chessboard = torch.stack(3 * [1 - x])
    return chessboard



input_image = checkerboard((128, 128), 16).to(torch.float32)
batch = torch.stack([input_image, input_image], dim=0)
# INN types:
layers = ['nice', 'all', 'nvp', 'glow', 'gin', 'affine']


class IUnetBlockTest(unittest.TestCase):

    def test_block_class(self):
        # Compare Layers of the CNNs:
        for c_l in layers:
            block = IUnetCOnvBlock((3, 16, 16), depth=10, coupling_layer=c_l)
            forward_jac = block(batch)
            forward = forward_jac[0][0]
            reverse = block(forward, rev=True)[0][0]
            self.assertTrue(torch.mean(torch.abs(batch - reverse)) < 1e-3,
                            msg="Reversed image does not match with input")
            print(f"model with {c_l} layer")

    def test_unet_class(self):
        for c_l in layers:
            model = iUNet(
                in_channels=3,
                channels=(3, 3, 8, 128),
                architecture=(2, 2, 2, 2),
                conv_depth=2,
                coupling_layer=c_l,
                dim=2
            )
            forward = model(batch)[0]
            reverse = model(forward, rev=True)[0]

            print(f"model with {c_l} layer")
            del model
