/*********************************************************************
* Filename:   base64.h
* Author:     Brad Conte (brad AT bradconte.com)
* Modified by: Gustavo Estrela and Bruno Sesso
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding Base64 implementati
*             on.
*********************************************************************/
#ifndef BASE64_CU_H
#define BASE64_CU_H


#include <stddef.h>
#include <stdlib.h>
#include "../utils/cudaUtils.h"
#include "../utils/utils.h"

typedef unsigned char BYTE;             // 8-bit byte

// Returns the size of the output. If called with out = NULL, will just
// return the size of what the output would have been (without a 
// terminating NULL).
size_t base64_encode(const BYTE in[], BYTE out[], size_t len, 
        int newline_flag);

// Returns the size of the output. If called with out = NULL, will just 
// return the size of what the output would have been (without a 
// terminating NULL).
size_t base64_decode(const BYTE in[], BYTE out[], size_t len);



#endif   // BASE64_CU_H
