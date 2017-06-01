#import "cudaUtils.h"

void checkCudaErr() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro na chamada cuda: ");
        printf("%s\n", cudaGetErrorString(err));
    }
}