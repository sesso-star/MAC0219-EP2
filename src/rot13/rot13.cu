/*********************************************************************
* Filename:   rot-13.c
* Author:     Brad Conte (brad AT bradconte.com)
* Modified by: Gustavo Estrela and Bruno Sesso
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the ROT-13 encryption algorithm.
      Algorithm specification can be found here:
       *
      This implementation uses little endian byte order.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <cuda_runtime.h>

extern "C" {
#include "rot13_cu.h"
}

/*********************** FUNCTION DEFINITIONS ***********************/
__global__ void rot13Kernel(char str[], int numElements)
{
    int case_type;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < numElements) {
        // Only process alphabetic characters.
        if (!(str[idx] < 'A' || (str[idx] > 'Z' && str[idx] < 'a') || str[idx] > 'z')) {
            // Determine if the char is upper or lower case.
            if (str[idx] >= 'a')
                case_type = 'a';
            else
                case_type = 'A';
            // Rotate the char's value, ensuring it doesn't accidentally "fall off" the end.
            str[idx] = (str[idx] + 13) % (case_type + 26);
            if (str[idx] < 26)
                str[idx] += case_type;
        }
    }
}

void rot13(char h_text[]) {
    int numElements = strlen(h_text);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(char);
    /*printf("string of %d elements\n", numElements);*/

    // Allocate the device input vector A
    char *d_text = NULL;
    err = cudaMalloc((void **)&d_text, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    /*printf("Copy input data from the host memory to the CUDA device\n");*/
    err = cudaMemcpy(d_text, h_text, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy string from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = nWarps * 32;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    /*printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);*/

    rot13Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_text, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch rot13 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    /*printf("Copy output data from the CUDA device to the host memory\n");*/
    err = cudaMemcpy(h_text, d_text, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy string from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_text);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

