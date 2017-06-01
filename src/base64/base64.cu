/*********************************************************************
* Filename:   base64.cu
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the Base64 encoding algorithm.
*********************************************************************/

extern "C" {
    #include "base64_cu.h"
}

#define NEWLINE_INVL 76

/* Variables */
// Note: To change the charset to a URL encoding, replace the '+' 
// and '/' with '*' and '-'
static const BYTE charset[] = 
    {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};

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
    int block_size = 256;
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
    cudaMemcpy(cuda_charset, charset, 64 * sizeof(BYTE), 
            cudaMemcpyHostToDevice);
    checkCudaErr();

    /* Process text from idx 0 to 3 * blks */
    cuda_encode<<<blks / block_size + 1, block_size>>>
        (blks, cuda_in, cuda_charset, cuda_out, newline_flag);
    checkCudaErr();
    cudaMemcpy(out, cuda_out, len2 * sizeof(BYTE),
            cudaMemcpyDeviceToHost);
    checkCudaErr();
    /* Process the ramainder part */
    blk_ceiling = blks * 3;
    len2 = blks * 4; 
    if (newline_flag)
        len2 += blks * 4 / (NEWLINE_INVL + 1);
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
    cudaFree(cuda_charset);
    return(len2);
}


__device__
int next_in_idx(const BYTE *cuda_in, int idx) {
    idx++;
    if (cuda_in[idx] == '\n') 
        idx++;
    return idx;
}


__global__
void cuda_decode(size_t len, const BYTE *cuda_in, BYTE *cuda_out,
        BYTE *cuda_revchar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int m = NEWLINE_INVL + 1;
        int new_lines = i * 4 / m;
        int i_in = next_in_idx(cuda_in, i * 4 + new_lines - 1);
        int i_out = i * 3;
        int a, b;

        a = cuda_revchar[cuda_in[i_in]] << 2; 
        i_in = next_in_idx(cuda_in, i_in);
        b = (cuda_revchar[cuda_in[i_in]] & 0x30) >> 4;
        cuda_out[i_out++] = a | b;
    
        a = cuda_revchar[cuda_in[i_in]] << 4;
        i_in = next_in_idx(cuda_in, i_in);
        b = cuda_revchar[cuda_in[i_in]] >> 2;
        cuda_out[i_out++] = a | b;

        a = cuda_revchar[cuda_in[i_in]] << 6;
        i_in = next_in_idx(cuda_in, i_in);
        b = cuda_revchar[cuda_in[i_in]];
        cuda_out[i_out] = a | b;
    }
}


size_t base64_decode(const BYTE in[], BYTE out[], size_t len) {
    size_t len2, blks, blk_ceiling, left_over;
    size_t no_newline_len, no_newline_leftover;
    int i, newline_flag;
    int block_size = 256;
    BYTE *revchar = (BYTE *) malloc(256 * sizeof(BYTE));
    BYTE *cuda_revchar, *cuda_in, *cuda_out;

    for (i = 0; i < 256; i++)
        revchar[i] = 'A';
    for (i = 0; i < 64; i++)
        revchar[charset[i]] = i;
    
    /* Calculates output length */
    if (in[len - 1] == '=') len--;
    if (in[len - 1] == '=') len--;
    newline_flag = len >= 77 && in[NEWLINE_INVL] == '\n';
    no_newline_len = len;
    if (newline_flag)
        no_newline_len -= len / (NEWLINE_INVL + 1);
    len2 = (no_newline_len / 4) * 3;
    no_newline_leftover = no_newline_len % 4;
    if (no_newline_leftover > 1)
        len2 += no_newline_leftover - 1;
    
    if (out == NULL) 
        return len2;
 
    blks = len / 4;
    left_over = len % 4;
    cudaMalloc(&cuda_in, sizeof(BYTE) * len);
    checkCudaErr();
    cudaMalloc(&cuda_out, sizeof(BYTE) * len2);
    checkCudaErr();
    cudaMalloc(&cuda_revchar, sizeof(BYTE) * 256);
    checkCudaErr();
    cudaMemcpy(cuda_in, in, len * sizeof(BYTE),
            cudaMemcpyHostToDevice);
    checkCudaErr();
    cudaMemcpy(cuda_revchar, revchar, 256 * sizeof(BYTE), 
            cudaMemcpyHostToDevice);
    checkCudaErr();

    /* Process idx from 0 to 4 * blks */
    cuda_decode<<<blks / block_size + 1, block_size>>>(blks,
            cuda_in, cuda_out, cuda_revchar);
    checkCudaErr();
    cudaMemcpy(out, cuda_out, len2 * sizeof(BYTE),
            cudaMemcpyDeviceToHost);
    checkCudaErr();
    /* Process the remainder part */
    blk_ceiling = blks * 4;
    if (newline_flag)
        len2 = 3 * (blk_ceiling - blk_ceiling / (NEWLINE_INVL + 1)) / 4;
    else
        len2 = 3 * blks;


    if (left_over == 2) 
        out[len2++] = (revchar[in[blk_ceiling]] << 2) |
            ((revchar[in[blk_ceiling + 1]] & 0x30) >> 4);
    else if (left_over == 3) {
        out[len2++] = (revchar[in[blk_ceiling]] << 2) |
            ((revchar[in[blk_ceiling + 1]] & 0x30) >> 4);
        out[len2++] = (revchar[in[blk_ceiling + 1]] << 4) |
            (revchar[in[blk_ceiling + 2]] >> 2);
    }
    out[len2] = '\0';
    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_revchar);
    return len2;
}
