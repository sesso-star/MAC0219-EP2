/*********************************************************************
* Filename:   base64_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding Base64
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "base64_cu.h"
#include "../utils/utils.h"

int nWarps;


// Just encodes a text using base64 algorithms
void testFileTime(char *filename) {
    char *text = NULL;
    int len = readTextFile(filename, &text);
    char *encoded_text = malloc(len * sizeof(char));
    base64_encode((const BYTE *) text, (BYTE *) encoded_text, len, 0);
    printf("Done\n");
}


/*********************** FUNCTION DEFINITIONS ***********************/
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf ("Usage: base64_test <number of threads wanted / 32>");
        return -1;
    }
    char *filename = argv[1];
    sscanf(argv[2], "%d", &nWarps);

    printf ("Will start test\n");
    testFileTime(filename);
    return 0;
}
