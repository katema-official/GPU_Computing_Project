objects = main.o GPU_exhaustive_opt_smem.o GPU_exhaustive_opt_common.o

config = ../config.h ../../../common.h

main.o: $(config)

compile_main: $(objects)
	nvcc -arch=sm_37 $(objects) -o main

%.o: %.cu
	nvcc -x cu -arch=sm_37 -I. -dc $< -o $@

exec_main_0:
	./main 1 36 100000 1 1048576

exec_main_1:
	./main 1 100 1000000000 1

exec_main_2:
	./main 1 100 10000000000 1

clean:
	rm -f *.o main

#https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/