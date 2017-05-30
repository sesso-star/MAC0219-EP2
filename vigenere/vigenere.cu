/*********************************************************************
* Filename:    vigenere.c
* Author:      programmingalgorithms.com
* Modified by: Gustavo Estrela and Bruno Sesso
* Copyright:
* Disclaimer:  This code is presented "as is" without any guarantees.
* Details:     Code available at 
*              https://www.programmingalgorithms.com/algorithm/
*              vigenere-cipher?lang=C
*********************************************************************/

extern "C" {
#include "vigenere.h"
}


void check_cuda_error() 
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro na chamada cuda: ");
        printf("%s\n", cudaGetErrorString(err));
    }
}


__device__
int mod(int a, int b)
{
    return (a % b + b) % b;
}


__device__
int is_alpha(char c)
{
    if (c >= 'A' && c <= 'Z' ||
        c >= 'a' && c <= 'z')
        return 1;
    else
        return 0;
}


__device__
int is_upper(char c)
{
    if (c <= 'Z')
        return 1;
    else
        return 0;
}


__device__
char to_upper(char c)
{
    if ('a' <= c && c <= 'z')
        return 'A' + c - 'a';
    else
        return c;
}


__device__
char to_lower(char c)
{
    if ('A' <= c && c <= 'Z')
        return 'a' + c - 'A';
    else
        return c;
}


__global__
void cipher(char *input, char *output, char *key, int encipher, int len,
        int key_len) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    if (is_alpha(input[i]))
    {
        int c_is_upper = is_upper(input[i]);
        char offset = c_is_upper ? 'A' : 'a';
        int key_index = i % key_len;
        char k;
        if (c_is_upper)
            k = to_upper(key[key_index]);
        else
            k = to_lower(key[key_index]) - offset;
        k = encipher ? k : -k;
        char ch = (char)((mod(((input[i] + k) - offset), 26)) + offset);
        output[i] = ch;
    }
    else
        output[i] = input[i];
}


void cipher(char *input, char *output, char *key, int encipher)
{
    char *cuda_in, *cuda_out, *cuda_key;
    int key_len = strlen(key);
    int len = strlen(input);
    int block_size = 256;
    
    cudaMalloc(&cuda_in, sizeof(char) * len);
    check_cuda_error();
    cudaMalloc(&cuda_out, sizeof(char) * len);
    check_cuda_error();
    cudaMalloc(&cuda_key, sizeof(char) * key_len);
    check_cuda_error();
    cudaMemcpy(cuda_in, input, len * sizeof(char),
            cudaMemcpyHostToDevice);
    check_cuda_error();
    cudaMemcpy(cuda_key, key, key_len * sizeof(char),
            cudaMemcpyHostToDevice);
    check_cuda_error();

    cipher<<<len / block_size + 1, block_size>>> 
        (cuda_in, cuda_out, cuda_key, encipher, len, key_len);
    check_cuda_error();
    cudaMemcpy(output, cuda_out, len * sizeof(char), 
            cudaMemcpyDeviceToHost);
    check_cuda_error();

    output[len] = '\0';
    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_key);
}


int encipher(char *input, char *output, char *key)
{
    cipher(input, output, key, 1);
    /*printf("|%s|\n", output);*/
    /*printf("output len: %d\n", strlen(output));*/
    output[strlen(output)] = '\0';
    return strlen(output);
}


int decipher(char *input, char *output, char *key)
{
    cipher(input, output, key, 0);
    /*printf ("|%s|\n", output);*/
    /*printf("output len: %d\n", strlen(output));*/
    output[strlen(output)] = '\0';
    return strlen(output);
}        
