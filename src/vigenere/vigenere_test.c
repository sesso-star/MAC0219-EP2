/*********************************************************************
* Filename: vigenere_test.c
* Author: Bruno Sesso and Gustavo Estrela
*********************************************************************/


/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "vigenere_cu.h"
#include "../utils/utils.h"

int nWarps;


/*********************** FUNCTION DEFINITIONS ***********************/

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf ("Usage: vigenere_test.c <number of threads wanted\
                / 32>");
        return -1;
    }
    char *filename = argv[1];
    sscanf(argv[2], "%d", &nWarps);

    printf("Vigenere tests: %s\n", vigenere_test() ? 
            "PASSED" : "FAILED");
    return 0;
}
