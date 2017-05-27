SERVER=gestrela@shell.vision.ime.usp.br:~/nvidia

base64_test_cuda: base64_test.o base64_cu.o
	nvcc -g base64_cu.o base64_test.o -o base64_test_cuda
	scp base64_test_cuda $(SERVER)

base64_test_seq: base64_test.o base64.o
	gcc base64_test.o base64.o -o base64_test_seq 

base64_test.o: base64_test.c
	gcc -c base64_test.c -o base64_test.o

base64_cu.o: base64.cu
	nvcc -g -c base64.cu -o base64_cu.o

base64.o: base64.c
	gcc -c base64.c -o base64.o

clean:
	rm -f base64_test_cuda base64_test_seq *.o
