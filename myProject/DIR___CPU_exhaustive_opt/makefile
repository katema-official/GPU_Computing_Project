compile_main: main.cu CPU_exhaustive_opt.cu
	nvcc -arch=sm_37 main.cu CPU_exhaustive_opt.cu -o main

exec_main_0:
	./main 1 1 36 100000 1

exec_main_1:
	./main 0 1 100 1000000000 1

exec_main_2:
	./main 0 1 100 10000000000 1