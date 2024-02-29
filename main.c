#include "trt_connector.h"
#include <cstdio>

int main(int argc, char *argv[]) {
    printf("Hello, World!\n");
    if (argc < 3) {
        printf( "Usage: %s <path to weights file>  <path to save engine> \n", argv[0]);
        return -1;
    }

    handle handle_trt = createTRT(argv);
    buildTRT(handle_trt);
    executeTRT(handle_trt);
    destoryTRT(handle_trt);
    return 0;
}