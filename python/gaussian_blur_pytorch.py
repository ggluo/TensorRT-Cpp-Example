import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import matplotlib.pyplot as plt
import sys

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.create_gaussian_kernel()

    def create_gaussian_kernel(self):
        x = np.linspace(-self.padding, self.padding, self.kernel_size)
        y = np.linspace(-self.padding, self.padding, self.kernel_size)
        x, y = np.meshgrid(x, y)
        kernel = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        self.kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        x = x.permute(3, 2, 0, 1)
        x_0 = F.conv2d(x[:, 0, ...], self.kernel, padding=self.padding, groups=1)
        x_1 = F.conv2d(x[:, 1, ...], self.kernel, padding=self.padding, groups=1)
        return torch.stack([x_0, x_1], dim=1).permute(2, 3, 1, 0)

def main(cfl, img_path, onnx_path):

    input_tensor = utils.readcfl(cfl)
    input_tensor = torch.tensor(utils.cplx2float(input_tensor))[..., None]

    # Create a Gaussian blur filter with kernel size 5 and sigma 1.5
    gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.5)
    blurred_image = gaussian_blur(input_tensor)

    plt.imshow(abs(utils.float2cplx(blurred_image[..., 0])), cmap="gray")
    plt.savefig(img_path)


    # Export the model to ONNX
    torch.onnx.export(gaussian_blur, input_tensor, onnx_path, verbose=True)

    print("Input image shape:", input_tensor.shape)
    print("Blurred image shape:", blurred_image.shape)


if __name__ == "__main__":
    # parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python gaussian_blur_jax.py <cfl> <save_img_path> <onnx_path>")
        sys.exit(1)

    cfl = sys.argv[1]
    img_path = sys.argv[2]
    onnx_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2], sys.argv[3])
