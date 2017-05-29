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


void cipher(char *input, char *output, char *key, int encipher)
{
    int keyLen = strlen(key);

    int inputLen = strlen(input);
    int nonAlphaCharCount = 0;

    for (int i = 0; i < inputLen; ++i)
    {
        if (isalpha(input[i]))
        {
            int cIsUpper = isupper(input[i]);
            char offset = cIsUpper ? 'A' : 'a';
            int keyIndex = i % keyLen;
            int k = (cIsUpper ? toupper(key[keyIndex]) : tolower(key[keyIndex])) - offset;

            k = encipher ? k : -k;
            char ch = (char)((mod(((input[i] + k) - offset), 26)) + offset);
            output[i] = ch;
        }
        else
        {
            output[i] = input[i];
        }
    }

    output[inputLen] = '\0';
}


int encipher(char *input, char *output, char *key)
{
    cipher(input, output, key, 1);
    return strlen(output);
}


int decipher(char *input, char *output, char *key)
{
    cipher(input, output, key, 0);
    return strlen(output);
}        
