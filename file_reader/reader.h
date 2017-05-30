#ifndef READER
#define READER

#include <sys/stat.h>
#include <unistd.h>

typedef unsigned char BYTE;

BYTE *read_file(char *filename);

#endif
