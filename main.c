#include "trt_connector.h"
#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]) {
    printf("Hello, World!\n");
    if (argc < 3) {
        printf( "Usage: %s <path to weights file>  <path to save engine> \n", argv[0]);
        return -1;
    }

    handle handle_trt = createTRT(argv);

    std::vector<float> input={0.5f, -0.5f, 1.0f};
    std::vector<float> output(2, 0.0f);

    buildTRT(handle_trt);
    executeTRT(handle_trt, input, 0, output, 1);
    std::cout << "Output: " << output[0] << ", " << output[1] << std::endl;
    destoryTRT(handle_trt);
    return 0;
}