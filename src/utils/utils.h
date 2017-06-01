#ifndef UTILS_H
#define UTILS_H

/*************************** HEADER FILES ***************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*********************** FUNCTION DECLARATIONS **********************/
int readTextFile(char* filename, char* str[]);
void checkCudaErr();

extern int nWarps;

#endif  
