#include "utils.h"

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

