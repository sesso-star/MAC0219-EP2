#include <stdio.h>
#include "../utils/utils.h"

int main(int argc, char* argv[]) {
    char* str = NULL;
    readTextFile(argv[1], &str);

    return 0;
}
