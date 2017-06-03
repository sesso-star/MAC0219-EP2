/*********************************************************************
* Filename: vigenere_test.c
* Author: Bruno Sesso and Gustavo Estrela
*********************************************************************/


/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "vigenere.h"
#include "../utils/utils.h"

int nWarps;


// Just enciphers a text using vigenere algorithms
void testFileTime(char *filename) {
    char *text = NULL;
    char *key = "some_key";
    int len = readTextFile(filename, &text);
    char *enciphered_text = malloc(len * sizeof(char));
    encipher(text, enciphered_text, key);
    printf("Done\n");
}


/*********************** FUNCTION DEFINITIONS ***********************/

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf ("Usage: vigenere_test <filename>\
                <number of threads wanted / 32>");
        return -1;
    }
    char *filename = argv[1];
    sscanf(argv[2], "%d", &nWarps);

    printf ("Will start test\n");
    testFileTime(filename);
    return 0;
}
