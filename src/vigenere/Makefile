SERVER=gestrela@shell.vision.ime.usp.br:~/nvidia

vigenere_test_cuda: vigenere_test.o vigenere_cu.o
	nvcc -g vigenere_cu.o vigenere_test.o -o vigenere_test_cuda
	scp vigenere_test_cuda $(SERVER)

vigenere_test_seq: vigenere_test.o vigenere.o
	gcc vigenere_test.o vigenere.o -o vigenere_test_seq 

vigenere_test.o: vigenere_test.c
	gcc -c vigenere_test.c -o vigenere_test.o

vigenere_cu.o: vigenere.cu
	nvcc -g -c vigenere.cu -o vigenere_cu.o

vigenere.o: vigenere.c
	gcc -g -c vigenere.c -o vigenere.o

clean:
	rm -f vigenere_test_cuda vigenere_test_seq *.o
