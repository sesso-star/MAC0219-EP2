/*********************************************************************
* Filename:   base64_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding Base64
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "../base64/base64.h"
#include "../vigenere/vigenere.h"
#include "../rot13/rot13.h"
#include "../utils/utils.h"

int nWarps;

/*********************** FUNCTION DEFINITIONS ***********************/
int base64Test()
{
    BYTE text[3][1024] = {{"fo"},
                          {"foobar"},
                          {"Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure."}};
    BYTE code[3][1024] = {{"Zm8="},
                          {"Zm9vYmFy"},
                          {"TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlz\nIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2Yg\ndGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGlu\ndWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRo\nZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4="}};
    BYTE buf[1024];
    size_t buf_len;
    int pass = 1;
    int idx;

    for (idx = 0; idx < 3; idx++) {
        buf_len = base64_encode(text[idx], buf, strlen(text[idx]), 1);
        pass = pass && ((buf_len == strlen(code[idx])) &&
                         (buf_len == base64_encode(text[idx], NULL, strlen(text[idx]), 1)));
        pass = pass && !strcmp(code[idx], buf);
        /*printf("Passed encode: %d\n", pass);*/

        memset(buf, 0, sizeof(buf));
        buf_len = base64_decode(code[idx], buf, strlen(code[idx]));
        pass = pass && ((buf_len == strlen(text[idx])) &&
                        (buf_len == base64_decode(code[idx], NULL, strlen(code[idx]))));
        pass = pass && !strcmp(text[idx], buf);
        /*printf("Passed decode: %d\n", pass);*/

    }
    return(pass);
}


int rot13Test() {
    char h_text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    char h_text_copy[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    char h_code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};
    int pass = 1;
    rot13(h_text);
    pass && !strcmp(h_text, h_code);
    rot13(h_text);
    pass && !strcmp(h_text_copy, h_text);
    return pass;
}


int vigenereTest()
{
    char text[3][1024] = {{"fo"},
                          {"foobar"},
                          {"Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure."}};
    char ciphered[3][1024] = {{"za"},
                              {"zaoocy"},
                              {"Gmn kz ymmfiaibinlyp, pvt shxy df cmm rrczoi, nug iy xbus upnbyfmr rhsnmiz stvm sntee hndquxs, dhdgb if h gymf bh tci yiaf, oluf oa a tydsrxlrvrwq bh dzpcshg pn xbq pqutdroqd cud mhpescaibevxe ilnzvufibp oa ezojnldbi, ekeleyw fhr zhjvn vrjlmzrwq bh aic oaephl tfqafwye."}};

    char key[256] = "uma chave";
    char buf[1024];
    int buf_len;
    int pass = 1;
    int idx;

    for (idx = 0; idx < 2; idx++) {
        buf_len = encipher(text[idx], buf, key);
        pass = pass && (buf_len == strlen(ciphered[idx]));
        pass = pass && !strcmp(ciphered[idx], buf);
        /*printf("Passed encipher: %d\n", pass);*/

        memset(buf, 0, sizeof(buf));
        buf_len = decipher(ciphered[idx], buf, key);
        pass = pass && (buf_len == strlen(text[idx]));
        pass = pass && !strcmp(text[idx], buf);
        /*printf("Passed decode: %d\n", pass);*/
    }

    return(pass);
}



void checkResult(int passed) {
    if (passed)
        printf("[OK]\n");
    else
        printf("[FAIL]\n");
}


int main(int argc, char *argv) {
    nWarps = 32;

    printf("Testing base64...");
    checkResult(base64Test());

    printf("Testing rot13...");
    checkResult(rot13Test());

    printf("Testing vigenere...");
    checkResult(vigenereTest());
}
