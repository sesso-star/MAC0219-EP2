/*********************************************************************
* Filename:    vigenere.c
* Author:      programmingalgorithms.com
* Modified by: Gustavo Estrela and Bruno Sesso
* Copyright:
* Disclaimer:  This code is presented "as is" without any guarantees.
* Details:     Code available at 
*              https://www.programmingalgorithms.com/algorithm/
*              vigenere-cipher?lang=C
*********************************************************************/

#include "vigenere.h"

int mod(int a, int b)
{
    return (a % b + b) % b;
}


char *cipher(char *input, char *key, int encipher)
{
    int keyLen = strlen(key);

    /*for (int i = 0; i < keyLen; ++i)*/
        /*if (!isalpha(key[i]))*/
            /*return ""; // Error*/

    int inputLen = strlen(input);
    char *output = (char *)malloc(inputLen + 1);
    int nonAlphaCharCount = 0;

    for (int i = 0; i < inputLen; ++i)
    {
        if (isalpha(input[i]))
        {
            int cIsUpper = isupper(input[i]);
            char offset = cIsUpper ? 'A' : 'a';
            int keyIndex = (i - nonAlphaCharCount) % keyLen;
            int k = (cIsUpper ? toupper(key[keyIndex]) : tolower(key[keyIndex])) - offset;
            k = encipher ? k : -k;
            char ch = (char)((mod(((input[i] + k) - offset), 26)) + offset);
            output[i] = ch;
        }
        else
        {
            output[i] = input[i];
            ++nonAlphaCharCount;
        }
    }

    output[inputLen] = '\0';
    return output;
}


int encipher(char *input, char *output, char *key)
{
    output = cipher(input, key, 1);
    printf ("%s\n", output);
    return strlen(output);
}


int decipher(char *input, char *output, char *key)
{
    output = cipher(input, key, 0);
    printf ("%s\n", output);
    return strlen(output);
}        
