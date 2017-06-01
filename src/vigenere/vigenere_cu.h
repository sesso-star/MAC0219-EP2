/*********************************************************************
* Filename:   vigenere.h
* Author:     programmingalgorithms.com
* Modified by: Gustavo Estrela and Bruno Sesso
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Code available at 
*             https://www.programmingalgorithms.com/algorithm/
*             vigenere-cipher?lang=C
*********************************************************************/

#ifndef VIGENERE_CU_H
#define VIGENERE_CU_H

#include <string.h>
#include <stddef.h>
#include <ctype.h>
#include "../utils/cudaUtils.h"


// Given a key and a string input, ciphers the input
//
int encipher(char *input, char *output, char *key);


// Given a ciphered text and a key, deciphers the text
//
int decipher(char *input, char *output, char *key);


#endif   // VIGENERE_CU_H
