import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tf2onnx
import sys
import utils
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# referred to https://github.com/onnx/tensorflow-onnx/tree/main

def gaussian_kernel(kernel_size, sigma):
    k = (kernel_size - 1) // 2
    x = np.arange(-k, k + 1)
    y = np.arange(-k, k + 1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel[..., None, None]

class GaussianBlur(tf.Module):
    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.kernel = tf.constant(gaussian_kernel(kernel_size, sigma), dtype=tf.float32)

    def __call__(self, input):
        input = tf.transpose(input, (3, 1, 0, 2))
        x_0 = tf.nn.conv2d(input[..., 0][..., None], self.kernel, strides=[1, 1], padding='SAME')
        x_1 = tf.nn.conv2d(input[..., 1][..., None], self.kernel, strides=[1, 1], padding='SAME')
        out = tf.concat([x_0, x_1], axis=-1)
        return tf.transpose(out, (2, 1, 3, 0), name='output') # be careful with the F-ordering and the name``


def main(cfl, img_path, onnx_path):

    with tf.device("/cpu:0"):
        
        cfl_array = utils.cplx2float(utils.readcfl(cfl))
        input_image = cfl_array[..., None]  # be careful with the F-ordering

        # Create a Gaussian blur filter with kernel size 5 and sigma 1.5
        module = GaussianBlur(kernel_size=5, sigma=1.5)
        blurred_image = module(input_image)

        plt.imshow(abs(utils.float2cplx(blurred_image.numpy().squeeze())), cmap="gray")
        plt.savefig(img_path)

        # customizing input name
        tf_func = tf.function(lambda input: module(input))
        tf_func = tf_func.get_concrete_function(tf.TensorSpec(input_image.shape, input_image.dtype))
        frozen_func = convert_variables_to_constants_v2(tf_func)
        graph_def = frozen_func.graph.as_graph_def()

        model_proto, _ = tf2onnx.convert.from_graph_def(graph_def, input_names=["input:0"], output_names=["output:0"])
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        #tf.io.write_graph(graph_def, "gaussian_blur_model", "gaussian_blur_model.pb", as_text=True)

        print("Input image shape:", input_image.shape)
        print("Blurred image shape:", blurred_image.shape)

        assert input_image.shape == blurred_image.shape


if __name__ == "__main__":
    # parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python gaussian_blur_jax.py <cfl> <save_img_path> <onnx_path>")
        sys.exit(1)

    cfl = sys.argv[1]
    img_path = sys.argv[2]
    onnx_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2], sys.argv[3])