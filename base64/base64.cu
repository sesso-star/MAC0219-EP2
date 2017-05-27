/*********************************************************************
* Filename:   base64.cu
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the Base64 encoding algorithm.
*********************************************************************/

#include <stdlib.h>
#include <stdio.h>

extern "C" {
    #include "base64.h"
}

#define NEWLINE_INVL 76

/* Variables */
// Note: To change the charset to a URL encoding, replace the '+' 
// and '/' with '*' and '-'
static const BYTE charset[] = 
    {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};


void checkCudaErr() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro na chamada cuda: ");
        printf("%s\n", cudaGetErrorString(err));
    }
}


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
void cuda_encode(size_t len, const BYTE *cuda_in, BYTE *cuda_charset, 
        BYTE *cuda_out, int new_line_flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int i_in = i * 3;
        int m = NEWLINE_INVL + 1;
        int new_lines = i * 4 / m;
        int i_out = i * 4 + new_lines;
        
        if (new_line_flag && i_out % m == m - 1)
            cuda_out[i_out++] = '\n';
        cuda_out[i_out++] = cuda_charset[cuda_in[i_in] >> 2];
       
        if (new_line_flag && i_out % m == m - 1)
            cuda_out[i_out++] = '\n';
        cuda_out[i_out++] = 
            cuda_charset[((cuda_in[i_in] & 0x03) << 4) |
                         (cuda_in[i_in + 1] >> 4)];

        if (new_line_flag && i_out % m == m - 1)
            cuda_out[i_out++] = '\n';
        cuda_out[i_out++] = 
            cuda_charset[((cuda_in[i_in + 1] & 0x0f) << 2) |
                         (cuda_in[i_in + 2] >> 6)];
        
        if (new_line_flag && i_out % m == m - 1)
            cuda_out[i_out++] = '\n';
        cuda_out[i_out++] = 
            cuda_charset[cuda_in[i_in + 2] & 0x3F];
    }
}


extern "C"
size_t base64_encode(const BYTE in[], BYTE out[], size_t len, 
        int newline_flag) {
    size_t blks, blk_ceiling, left_over, len2;
    int blockSize = 256;
    BYTE *cuda_in, *cuda_out, *cuda_charset;

    blks = (len / 3);
    left_over = len % 3;
    len2 = blks * 4;
    if (left_over)
        len2 += 4;
    if (newline_flag)
        len2 += len / 57;

    if (out == NULL) 
        return len2;
    
    cudaMalloc(&cuda_in, sizeof(BYTE) * len);
    checkCudaErr();
    cudaMalloc(&cuda_out, sizeof(BYTE) * len2);
    checkCudaErr();
    cudaMalloc(&cuda_charset, sizeof(BYTE) * 64);
    checkCudaErr();
    cudaMemcpy(cuda_in, in, len * sizeof(BYTE), 
            cudaMemcpyHostToDevice);
    checkCudaErr();
    cudaMemcpy(cuda_out, out, len * sizeof(BYTE), 
            cudaMemcpyHostToDevice);
    checkCudaErr();
    cudaMemcpy(cuda_charset, charset, 64 * sizeof(BYTE), 
            cudaMemcpyHostToDevice);
    checkCudaErr();

    /* Process text from idx 0 to 3 * blks */
    cuda_encode<<<blks / blockSize + 1, blockSize>>>
        (blks, cuda_in, cuda_charset, cuda_out, newline_flag);
    checkCudaErr();
    cudaMemcpy(out, cuda_out, len2 * sizeof(BYTE),
            cudaMemcpyDeviceToHost);
    checkCudaErr();
    /* Process the ramainder part */
    blk_ceiling = blks * 3;
    len2 = blks * 4 + (blks * 4 / (NEWLINE_INVL + 1));
    if (left_over == 1) {
        out[len2++] = charset[in[blk_ceiling] >> 2];
        out[len2++] = charset[(in[blk_ceiling] & 0x03) << 4];
        out[len2++] = '=';
        out[len2++] = '=';
    }
    else if (left_over == 2) {
        out[len2++] = charset[in[blk_ceiling] >> 2];
        out[len2++] = charset[((in[blk_ceiling] & 0x03) << 4) |
            (in[blk_ceiling + 1] >> 4)];
        out[len2++] = charset[(in[blk_ceiling + 1] & 0x0F) << 2];
        out[len2++] = '=';
    }
    out[len2] = '\0';

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    return(len2);
}


size_t base64_decode(const BYTE in[], BYTE out[], size_t len) {
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
