/*********************************************************************
* Filename: rot13_test.c
* Author: Bruno Sesso and Gustavo Estrela
*********************************************************************/


/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include "rot13_cu.h"
#include "../utils/utils.h"

int nWarps;

/*********************** FUNCTION DEFINITIONS ***********************/
void testWithString() {
   char h_text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
   char h_code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};

   rot13(h_text);

   if (strcmp(h_text, h_code)) {
      fprintf(stderr, "Result verification failed");
      exit(EXIT_FAILURE);
   }
   printf("Test PASSED\n");

   printf("Done\n");
}

void testWithFile(char* filename) {
   char* h_text = NULL;

   printf("Reading file: %s\n", filename);
   int numElements = readTextFile(filename, &h_text);
   size_t size = numElements * sizeof(char);

   // Allocate the device answer string
   char* original = (char*) malloc(size);
   original = strcpy(original, h_text);

   rot13(h_text);
   rot13(h_text);

   if (strcmp(original, h_text)) {
      fprintf(stderr, "Result verification failed");
      exit(EXIT_FAILURE);
   }

   printf("Test PASSED\n");
   printf("Done\n");
}

int main(int argc, char* argv[]) {
   char* filename = argv[1];
   sscanf(argv[2], "%d", &nWarps);

   printf("Will start test\n");
   testWithFile(filename);
   testWithString();

   return 0;
}