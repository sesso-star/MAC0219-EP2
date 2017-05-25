SERVER=gestrela@shell.vision.ime.usp.br:~/nvidia

base64_cu: base64.cu
	nvcc -o base64_cu base64.cu
	scp base64_cu $(SERVER)
