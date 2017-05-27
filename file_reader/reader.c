#include "reader.h"


BYTE *read_file(char *filename) {
    BYTE *data;
    int current_byte;
    FILE *file; 
    struct stat st;

    if (stat(filename, &st) == 0) {
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    }
    else {
        return NULL;
    }

    file = fopen(filename, "rb");
    current_byte = 0;
    while(fread(&data[current_byte], sizeof(BYTE), 1, file) == 1) {
        current_byte += 1;
    }

    return data;
}
