CC=gcc

all: test

test: rot-13_test

clean:
	rm -rf *.o

run:
	./rot-13_test

rot-13_test: rot-13_test.o rot-13.o
	$(CC) -o rot-13_test rot-13_test.o rot-13.o

rot-13_test.o: rot-13_test.c
	$(CC) -c rot-13_test.c -o rot-13_test.o

rot-13.o: rot-13.c 
	$(CC) -c rot-13.c -o rot-13.o
