import unittest

import cv2
import matplotlib.pyplot as plt
import torch

import FrEIA.framework as Ff
from FrEIA.iunets import iUNet, IUnetCOnvBlock


def compare_reverse(batch, forward, reverse, layer):
    plt.subplot(1, 3, 1)
    plt.imshow(batch.permute(1, 2, 0), vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.imshow(forward.detach().permute(1, 2, 0), vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.imshow(reverse.detach().permute(1, 2, 0), vmin=0, vmax=1)
    plt.title(f"model with  {layer}")
    plt.show()
    # print(f'TESTING BLOCK "{layer}"')


img = cv2.resize(cv2.imread('docs/freia_logo.png'), (128, 128)) / 255.0
input_image = torch.tensor(img).permute(2, 0, 1).type(torch.FloatTensor)
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
            compare_reverse(batch[0], forward[0], reverse[0], c_l)
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
            compare_reverse(batch[0], forward[0], reverse[0], c_l)
            del model



# if __name__ == "__main__":
#     unittest.main()

