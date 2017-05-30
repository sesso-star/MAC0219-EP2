/*********************************************************************
* Filename:   rot-13_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ROT-13
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "rot-13.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rot13_test()
{
    char text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    char code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};
    char buf[1024];
    int pass = 1;

    // To encode, just apply ROT-13.
    strcpy(buf, text);
    rot13(buf);
    pass = pass && !strcmp(code, buf);

    // To decode, just re-apply ROT-13.
    rot13(buf);
    pass = pass && !strcmp(text, buf);

    return(pass);
}

int readTextFile(char filename[], char* str[]) {
   long length;
   FILE * f = fopen (filename, "rb");

   if (f)
   {
     fseek (f, 0, SEEK_END);
     length = ftell (f);

     fseek (f, 0, SEEK_SET);
     *str = (char*) malloc(length);
     if (str)
     {
       fread (*str, 1, length, f);
     }
     fclose (f);
   }

   return length;
}

int main(int argc, char* argv[]) {
   char* h_text = NULL;
   char* filename = argv[1];

   printf("Reading file: %s\n", filename);

   // Print the vector length to be used, and compute its size
   int numElements = readTextFile(filename, &h_text);

   printf("File size: %d\n", numElements);

   size_t size = numElements * sizeof(char);

   char* buf = (char*) malloc(size);
   strcpy(buf, h_text);

   rot13(buf);
   rot13(buf);

   if (strcmp(buf, h_text)) {
      fprintf(stderr, "Result verification failed");
      exit(EXIT_FAILURE);
   }
   printf("Test PASSED\n");

   // Free host memory
   free(buf);

   printf("Done\n");
   return 0;
}

// int main()
// {
//     printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");

//     return(0);
// }
