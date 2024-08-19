#ifndef TRT_CONNECTOR_H
#define TRT_CONNECTOR_H

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* handle;

handle createTRT(char** argv);
void buildTRT(handle handle_t);
void destoryTRT(handle handle_t);
void executeTRT(handle handle_t, const std::vector<float>& input, int input_idx, std::vector<float>& output, int output_idx);

#ifdef __cplusplus
}
#endif

#endif 
