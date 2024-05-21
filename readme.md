# C++/C TensorRT Inference Example

This repository provides C++ and C examples that use TensorRT to inference the models that are implement with Pytorch/JAX/Tensorflow.

This is integrated into [BART](https://github.com/mrirecon/bart)

## Requirements

- Python 3.x
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)
- CUDA Toolkit
- PyTorch
- ONNX

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/ggluo/TensorRT-Tiny-Cpp-Example.git
    cd TensorRT-Tiny-Cpp-Example
    ```

2. Install onnx and torch if not:

    ```bash
    pip install torch onnx onnxscript
    ```

3. Ensure that [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) and CUDA Toolkit are installed on your system and specify it according in the makefile.
    ```makefile
    LDFLAGS = -L/path/to/TensorRT/lib
    INCLUDEDIRS = -I/path/to/TensorRT/include
    ```

## Running the Test

To run the test script, execute the following commands:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib
bash run_test.sh
```

This script performs the following steps:

1. Exports the ONNX model: `python data/export_model.py data/model.onnx`
2. Compiles the TensorRT inference code: `make`
3. Runs the TensorRT inference code: `./main data/model.onnx data/first_engine.trt`

The provided ONNX model is located at `data/model.onnx`, and the resulting TensorRT engine will be saved to `data/first_engine.trt`.

## Overview of `main.cpp`

The `main.cpp` file contains the main entry point for the TensorRT inference code. Below is an overview of its functionality:

```cpp
#include "trt.h"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Hello World from TensorRT" << std::endl;

    // Parse command-line arguments
    infer_params params{argv[1], 1,  argv[2], ""}; 

    // Initialize TensorRT inference object
    trt_infer trt(params);
    trt.build();

    // Copy input data from host to device
    trt.CopyFromHostToDevice({0.5f, -0.5f, 1.0f}, 0, nullptr);

    // Perform inference
    trt.infer();

    // Copy output data from device to host
    std::vector<float> output(2, 0.0f);
    trt.CopyFromDeviceToHost(output, 1, nullptr);

    // Print output
    std::cout << "Output: " << output[0] << ", " << output[1] << std::endl;

    return 0;
}
```

This code performs the following steps:

1. Initializes the TensorRT inference parameters using command-line arguments.
2. Initializes the TensorRT inference object and builds the inference engine.
3. Copies input data from the host to the device.
4. Performs inference.
5. Copies output data from the device to the host.
6. Prints the output.

## TODO
1. memory leakage check with valgrind
2. add c_connector
