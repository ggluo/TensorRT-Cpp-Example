#include "trt.h"
#include <iostream>
#include <cstring>
bool is_trt_available(char *path) {
    char *dot = strchr(path, '.');
    if (dot != NULL) 
        if (strcmp(dot, ".trt") == 0)
            return true;
    return false;
}

int main(int argc, char** argv) {
    std::cout << "Hello World from TensorRT" << std::endl;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path to weights file>  [path to save engine] " << std::endl;
        return 1;
    }

    bool load_engine = is_trt_available(argv[1]);
    
    auto params = std::make_unique<infer_params>(
        load_engine ? "" : argv[1], // Path to weights file if provided, empty string otherwise
        1,                         // Batch size (assuming a constant value of 1)
        argc>2 ? argv[2] : "",            // Path to save engine (last argument)
        load_engine ? argv[1] : ""  // Path to TRT file if provided, empty string otherwise
    );

    trt_infer trt(*params); // Added missing object initialization
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
