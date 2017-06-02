SERVER=gestrela@terra.eclipse.ime.usp.br:~/gustavo/MAC0219-EP2/bin

CC=gcc

OBJ_DIR=obj
BIN_DIR=bin

CFLAGS=-O0 -g
NVCCFLAGS=-Wno-deprecated-gpu-targets

B64_DIR=src/base64
UTILS_DIR=src/utils
VIG_DIR=src/vigenere
ROT13_DIR=src/rot13

all: base64 vigenere rot13

base64: $(BIN_DIR)/base64_test_cuda

vigenere: $(BIN_DIR)/vigenere_test_cuda

rot13: $(BIN_DIR)/rot13_test_cuda

rot13_test_seq: $(BIN_DIR)/rot13_test_seq

$(BIN_DIR)/base64_test_cuda: $(OBJ_DIR)/base64_test.o $(OBJ_DIR)/base64_cu.o $(OBJ_DIR)/cudaUtils.o $(OBJ_DIR)/utils.o
	nvcc $(NVCCFLAGS) $^ -o $@
	scp $@ $(SERVER)

$(BIN_DIR)/base64_test_seq: $(OBJ_DIR)/base64_test.o $(OBJ_DIR)/base64.o $(OBJ_DIR)/utils.o
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/base64_test.o: $(B64_DIR)/base64_test.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/base64_cu.o: $(B64_DIR)/base64.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/base64.o: $(B64_DIR)/base64.c
	$(CC) $(CFLAGS) -c $< -o $@



$(BIN_DIR)/vigenere_test_cuda: $(OBJ_DIR)/vigenere_test.o $(OBJ_DIR)/vigenere_cu.o $(OBJ_DIR)/cudaUtils.o $(OBJ_DIR)/utils.o
	nvcc $(NVCCFLAGS) $^ -o $@
	scp $@ $(SERVER)

$(BIN_DIR)/vigenere_test_seq: $(OBJ_DIR)/vigenere_test.o $(OBJ_DIR)/vigenere.o $(OBJ_DIR)/utils.o
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/vigenere_test.o: $(VIG_DIR)/vigenere_test.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/vigenere_cu.o: $(VIG_DIR)/vigenere.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/vigenere.o: $(VIG_DIR)/vigenere.c
	$(CC) $(CFLAGS) -c $< -o $@



$(BIN_DIR)/rot13_test_cuda: $(OBJ_DIR)/rot13_test.o $(OBJ_DIR)/rot13_cu.o $(OBJ_DIR)/utils.o 
	nvcc $(NVCCFLAGS) $^ -o $@

$(BIN_DIR)/rot13_test_seq: $(OBJ_DIR)/rot13_test.o $(OBJ_DIR)/rot13.o $(OBJ_DIR)/utils.o
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/rot13_test.o: $(ROT13_DIR)/rot13_test.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/rot13_cu.o: $(ROT13_DIR)/rot13.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/rot13.o: $(ROT13_DIR)/rot13.c
	$(CC) $(CFLAGS) -c $< -o $@






$(OBJ_DIR)/cudaUtils.o: $(UTILS_DIR)/cudaUtils.c
	nvcc $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/utils.o: $(UTILS_DIR)/utils.c
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(BIN_DIR)/base64_test_cuda $(BIN_DIR)/base64_test_seq $(OBJ_DIR)/*.o \
		  $(BIN_DIR)/vigenere_test_cuda $(BIN_DIR)/vigenere_test_seq \
		  $(BIN_DIR)/rot13_test_cuda $(BIN_DIR)/rot13_test_seq
