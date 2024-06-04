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
    std::cout << "===========Gaussian blur with TensorRT==============" << std::endl;
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <path to weights file>  <path to save engine> <input image> <output image>" << std::endl;
        return 1;
    }

    bool load_engine = is_trt_available(argv[1]);
    
    auto params = std::make_unique<infer_params>(
        load_engine ? "" : argv[1], // Path to weights file if provided, empty string otherwise
        1,                         // Batch size (assuming a constant value of 1)
        load_engine ? "" : argv[2],            // Path to save engine (last argument)
        load_engine ? argv[1] : ""  // Path to TRT file if provided, empty string otherwise
    );

    std::vector<int> dims = getDimsFromFile(argv[3]);
    std::vector<std::complex<float>> input = read_cfl(argv[3], dims);

    trt_infer trt(*params); // Added missing object initialization
    trt.build();

    std::vector<float> f_input(reinterpret_cast<float*>(input.data()),
                            reinterpret_cast<float*>(input.data()) + 2 * input.size());

    printf("==== inference without cudastream =====\n");

    trt.CopyFromHostToDevice(f_input, 0, nullptr);

    trt.infer();

    trt.CopyFromDeviceToHost(f_input, 1, nullptr);

    std::vector<std::complex<float>> output(reinterpret_cast<std::complex<float>*>(f_input.data()),
                    reinterpret_cast<std::complex<float>*>(f_input.data()) + f_input.size() / 2);

    write_cfl(argv[4], output, dims);
    
    return 0;
}
