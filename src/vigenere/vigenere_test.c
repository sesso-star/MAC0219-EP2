/*********************************************************************
* Filename: vigenere_test.c
* Author: Bruno Sesso and Gustavo Estrela
*********************************************************************/


/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "vigenere_cu.h"
#include "../utils/utils.h"

int nWarps;


/*********************** FUNCTION DEFINITIONS ***********************/
int vigenere_test()
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
        printf("Passed encipher: %d\n", pass);

        memset(buf, 0, sizeof(buf));
        buf_len = decipher(ciphered[idx], buf, key);
        pass = pass && (buf_len == strlen(text[idx]));
        pass = pass && !strcmp(text[idx], buf);
        printf("Passed decode: %d\n", pass);
    }

    return(pass);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf ("Usage: vigenere_test.c <number of threads wanted\
                / 32>");
        return -1;
    }
    char *filename = argv[1];
    sscanf(argv[2], "%d", &nWarps);

    printf("Vigenere tests: %s\n", vigenere_test() ? 
            "PASSED" : "FAILED");
    return 0;
}
