import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
import tensorflow as tf
import sys
from jax.experimental import jax2tf
import utils
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tf2onnx
import matplotlib.pyplot as plt


def apply_gaussian_blur(input, kernel_size=5, sigma=1.5):
    k = (kernel_size - 1) // 2
    x = jnp.arange(-k, k + 1)
    y = jnp.arange(-k, k + 1)
    x, y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    input = jnp.transpose(input, (3, 2, 0, 1))

    input_0 = jax.lax.conv(input[:, 0, ...][None, ...], kernel[None, None, ...], (1, 1), 'SAME')
    input_1 = jax.lax.conv(input[:, 1, ...][None, ...], kernel[None, None, ...], (1, 1), 'SAME')
    out = jnp.concatenate([input_0, input_1], axis=1)
    return jnp.transpose(out, (2, 3, 1, 0))


def main(cfl, img_path, onnx_path):

    # Convert JAX function to TensorFlow function
    tf_gaussian_blur = jax2tf.convert(lambda x: apply_gaussian_blur(x, kernel_size=5, sigma=1.5), enable_xla=False)

    
    cfl_array = utils.cplx2float(utils.readcfl(cfl))
    input_image = cfl_array[..., None]  # be careful with the F-ordering

    blurred_image = tf_gaussian_blur(tf.constant(input_image))

    plt.imshow(abs(utils.float2cplx(blurred_image.numpy().squeeze())), cmap="gray")
    plt.savefig(img_path)
    
    tf_func = tf.function(tf_gaussian_blur, autograph=False)
    """
    frozen_func =tf_func.get_concrete_function(tf.TensorSpec(input_image.shape, input_image.dtype))
    frozen_func = convert_variables_to_constants_v2(frozen_func)
    graph_def = frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_def, "./", "jax_gaussian.pb", as_text=True)
    """

    model_proto, _ = tf2onnx.convert.from_function(tf_func, (tf.TensorSpec(input_image.shape, input_image.dtype),))
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    assert blurred_image.shape == input_image.shape, "Output shape does not match input shape"

if __name__ == "__main__":
    # parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python gaussian_blur_jax.py <cfl> <save_img_path> <onnx_path>")
        sys.exit(1)

    cfl = sys.argv[1]
    img_path = sys.argv[2]
    onnx_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2], sys.argv[3])