#include "trt_connector.h"
#include "trt.h"

extern "C" {

handle createTRT(char** argv) {
    infer_params params{argv[1], 1,  argv[2], ""}; 
    return reinterpret_cast<handle>(new trt_infer(params));
}

void destoryTRT(handle handle_t) {
    delete reinterpret_cast<trt_infer*>(handle_t);
}

void buildTRT(handle handle_t) {
    reinterpret_cast<trt_infer*>(handle_t)->build();
}

void executeTRT(handle handle_t, const std::vector<float>& input, int input_idx, std::vector<float>& output, int output_idx) {
    trt_infer* trt = reinterpret_cast<trt_infer*>(handle_t);
    (trt)->CopyFromHostToDevice({0.5f, -0.5f, 1.0f}, input_idx, nullptr);
    (trt)->infer();
    (trt)->CopyFromDeviceToHost(output, output_idx, nullptr);
}

}
