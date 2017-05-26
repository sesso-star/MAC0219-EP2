/*********************************************************************
* Filename:   base64.cu
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the Base64 encoding algorithm.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <stdio.h>
extern "C" {
#include "base64.h"
}

/****************************** MACROS ******************************/
#define NEWLINE_INVL 76

/**************************** VARIABLES *****************************/
// Note: To change the charset to a URL encoding, replace the '+' and '/' with '*' and '-'
static const BYTE charset[]={"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};

void checkCudaErr(); 

/*********************** FUNCTION DEFINITIONS ***********************/
BYTE revchar(char ch)
{
    if (ch >= 'A' && ch <= 'Z')
        ch -= 'A';
    else if (ch >= 'a' && ch <='z')
        ch = ch - 'a' + 26;
    else if (ch >= '0' && ch <='9')
        ch = ch - '0' + 52;
    else if (ch == '+')
        ch = 62;
    else if (ch == '/')
        ch = 63;
    return(ch);
}


__global__
void cuda_encode(size_t len, const BYTE *cuda_in, BYTE *cuda_out, 
        int new_line_flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        cuda_out[i] = cuda_in[i];
    }
}


extern "C"
size_t base64_encode(const BYTE in[], BYTE out[], size_t len, 
        int newline_flag) {
    size_t idx, idx2, blks, blk_ceiling, left_over, newline_count = 0;
    int blockSize = 256, i;
    BYTE *cuda_in, *cuda_out;
    cudaError_t code;

    blks = (len / 3);
    left_over = len % 3;

    if (out == NULL) {
        idx2 = blks * 4 ;
        if (left_over)
            idx2 += 4;
        if (newline_flag)
            idx2 += len / 57;   
    }
    else {
        size_t len2 = len + blks + left_over;
        if (newline_flag)
            len2 += len / 57;

        cudaMalloc(&cuda_in, sizeof(BYTE) * len);
        checkCudaErr();
        cudaMalloc(&cuda_out, sizeof(BYTE) * len2);
        checkCudaErr();
        cudaMemcpy(cuda_in, in, len * sizeof(BYTE), 
                cudaMemcpyHostToDevice);
        checkCudaErr();
        for (i = 0; i < len; i++)
            out[i] = 0;
        
        blk_ceiling = blks * 3;
        cuda_encode<<<blk_ceiling / blockSize + 1, blockSize>>>
            (len, cuda_in, cuda_out, newline_flag);
        checkCudaErr();
        /*printf("%d blocks and %d threads on each block", */
                /*blk_ceiling / blockSize + 1, blockSize);*/
        cudaMemcpy(out, cuda_out, len2 * sizeof(BYTE), cudaMemcpyDeviceToHost);
        checkCudaErr();
        
        for (i = 0; i < len; i++) {
            printf("%c", out[i]);
        }
        printf("\n");
        /*for (idx = 0, idx2 = 0; idx < blk_ceiling; idx += 3, idx2 += 4) {*/
            /*out[idx2]     = charset[in[idx] >> 2];*/
            /*out[idx2 + 1] = charset[((in[idx] & 0x03) << 4) | (in[idx + 1] >> 4)];*/
            /*out[idx2 + 2] = charset[((in[idx + 1] & 0x0f) << 2) | (in[idx + 2] >> 6)];*/
            /*out[idx2 + 3] = charset[in[idx + 2] & 0x3F];*/
            // The offical standard requires a newline every 76 characters.
            // (Eg, first newline is character 77 of the output.)
            /*if (((idx2 - newline_count + 4) % NEWLINE_INVL == 0) && newline_flag) {*/
                /*out[idx2 + 4] = '\n';*/
                /*idx2++;*/
                /*newline_count++;*/
            /*}*/
        /*}*/

        /*if (left_over == 1) {*/
            /*out[idx2]     = charset[in[idx] >> 2];*/
            /*out[idx2 + 1] = charset[(in[idx] & 0x03) << 4];*/
            /*out[idx2 + 2] = '=';*/
            /*out[idx2 + 3] = '=';*/
            /*idx2 += 4;*/
        /*}*/
        /*else if (left_over == 2) {*/
            /*out[idx2]     = charset[in[idx] >> 2];*/
            /*out[idx2 + 1] = charset[((in[idx] & 0x03) << 4) | (in[idx + 1] >> 4)];*/
            /*out[idx2 + 2] = charset[(in[idx + 1] & 0x0F) << 2];*/
            /*out[idx2 + 3] = '=';*/
            /*idx2 += 4;*/
        /*}*/
    }

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    return(idx2);
}


void checkCudaErr() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro na chamada cuda: ");
        printf("%s\n", cudaGetErrorString(err));
    }
}


size_t base64_decode(const BYTE in[], BYTE out[], size_t len) {
    BYTE ch;
    size_t idx, idx2, blks, blk_ceiling, left_over;

    if (in[len - 1] == '=')
        len--;
    if (in[len - 1] == '=')
        len--;

    blks = len / 4;
    left_over = len % 4;

    if (out == NULL) {
        if (len >= 77 && in[NEWLINE_INVL] == '\n')   // Verify that newlines where used.
            len -= len / (NEWLINE_INVL + 1);
        blks = len / 4;
        left_over = len % 4;

        idx = blks * 3;
        if (left_over == 2)
            idx ++;
        else if (left_over == 3)
            idx += 2;
    }
    else {
        blk_ceiling = blks * 4;
        for (idx = 0, idx2 = 0; idx2 < blk_ceiling; idx += 3, idx2 += 4) {
            if (in[idx2] == '\n')
                idx2++;
            out[idx]     = (revchar(in[idx2]) << 2) | ((revchar(in[idx2 + 1]) & 0x30) >> 4);
            out[idx + 1] = (revchar(in[idx2 + 1]) << 4) | (revchar(in[idx2 + 2]) >> 2);
            out[idx + 2] = (revchar(in[idx2 + 2]) << 6) | revchar(in[idx2 + 3]);
        }

        if (left_over == 2) {
            out[idx]     = (revchar(in[idx2]) << 2) | ((revchar(in[idx2 + 1]) & 0x30) >> 4);
            idx++;
        }
        else if (left_over == 3) {
            out[idx]     = (revchar(in[idx2]) << 2) | ((revchar(in[idx2 + 1]) & 0x30) >> 4);
            out[idx + 1] = (revchar(in[idx2 + 1]) << 4) | (revchar(in[idx2 + 2]) >> 2);
            idx += 2;
        }
    }

    return(idx);
}
