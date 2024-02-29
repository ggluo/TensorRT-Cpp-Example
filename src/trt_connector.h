#ifndef TRT_CONNECTOR_H
#define TRT_CONNECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* handle;

handle createTRT(char** argv);
void buildTRT(handle handle_t);
void destoryTRT(handle handle_t);
void executeTRT(handle handle_t);

#ifdef __cplusplus
}
#endif

#endif 
