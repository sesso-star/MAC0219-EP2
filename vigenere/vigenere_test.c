/*********************************************************************
* Filename: vigenere_test.c
* Author: Bruno Sesso and Gustavo Estrela
*********************************************************************/


/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "vigenere.h"


/*********************** FUNCTION DEFINITIONS ***********************/
int vigenere_test()
{
    char text[3][1024] = {{"fo"},
                          {"foobar"},
                          {"Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure."}};
    char ciphered[3][1024] = {{""},
                              {""},
                              {""}};
    char cipher_key[256] = "uma chave";
    char buf[1024];
    int buf_len;
    int pass = 1;
    int idx;

    /*for (idx = 0; idx < 1; idx++) {*/
        /*buf_len = encipher(text[idx], buf, key);*/
        /*pass = pass && (buf_len == strlen(ciphered[idx]));*/
        /*pass = pass && !strcmp(code[idx], buf);*/
        /*printf("Passed encipher: %d\n", pass);*/

        /*memset(buf, 0, sizeof(buf));*/
        /*buf_len = decipher(ciphered[idx], buf, key);*/
        /*pass = pass && (buf_len == strlen(text[idx]));*/
        /*pass = pass && !strcmp(text[idx], buf);*/
        /*printf("Passed decode: %d\n", pass);*/

    /*}*/

    return(pass);
}

int main()
{
    char buf[1024];
    char text[1024] = "teste123 abc";
    char key[1024] = "chave";
    /*printf("Base64 tests: %s\n", base64_test() ? "PASSED" : "FAILED");*/
    encipher(text, buf, key);
    return 0;
}
