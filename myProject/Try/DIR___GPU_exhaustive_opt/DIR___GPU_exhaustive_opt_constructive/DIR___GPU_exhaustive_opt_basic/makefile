objects = main.o GPU_exhaustive_opt_basic.o ../GPU_exhaustive_opt_common.o

compile_main: $(objects)
	nvcc -arch=sm_37 $(objects) -o main

%.o: %.cu
	nvcc -x cu -arch=sm_37 -I. -dc $< -o $@

exec_main_0:
	./main 1 30 100000 1 32768

exec_main_1:
	./main 1 100 1000000000 1

exec_main_2:
	./main 1 100 10000000000 1

clean:
	rm -f *.o main

#https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/