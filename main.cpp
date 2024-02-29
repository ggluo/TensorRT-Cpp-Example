#include "trt.h"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Hello World from TensorRT" << std::endl;
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <path to weights file>  <path to save engine>" << std::endl;
        return 1;
    }
    infer_params params{argv[1], 1,  argv[2], ""}; 
    trt_infer trt(params); // Added missing object initialization
    trt.build();

    printf("==== inference without cudastream =====\n");

    trt.CopyFromHostToDevice({0.5f, -0.5f, 1.0f}, 0, nullptr);

    trt.infer();

    std::vector<float> output(2, 0.0f);
    trt.CopyFromDeviceToHost(output, 1, nullptr);

    std::cout << "Output: " << output[0] << ", " << output[1] << std::endl;

    printf("==== inference with cudastream =====\n");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trt.CopyFromHostToDevice({0.5f, -0.5f, 1.0f}, 0, stream);

    trt.infer();
    
    std::vector<float> output2(2, 0.0f);
    trt.CopyFromDeviceToHost(output2, 1, stream);
    cudaStreamDestroy(stream);

    std::cout << "Output: " << output2[0] << ", " << output2[1] << std::endl;
    
    return 0;
}
