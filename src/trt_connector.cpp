#include "trt_connector.h"
#include "trt.h"

extern "C" {

inline trt_infer* cast(handle handle_t) {
    return reinterpret_cast<trt_infer*>(handle_t);
}

handle createTRT(char** argv) {
    infer_params params{argv[1], 1,  argv[2], ""}; 
    return reinterpret_cast<handle>(new trt_infer(params));
}

void destoryTRT(handle handle_t) {
    delete cast(handle_t);
}

void buildTRT(handle handle_t) {
    cast(handle_t)->build();
}

void executeTRT(handle handle_t) {
    trt_infer* trt = cast(handle_t);
    (trt)->CopyFromHostToDevice({1.0f, 0.0f, 1.0f}, 0, nullptr);
    (trt)->infer();
    std::vector<float> output(2, 0.0f);
    (trt)->CopyFromDeviceToHost(output, 1, nullptr);
    std::cout << "Output: " << output[0] << ", " << output[1] << std::endl;
}

}
