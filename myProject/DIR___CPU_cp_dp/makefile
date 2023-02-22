compile_main: main.cu CPU_dp.cu
	nvcc -arch=sm_37 main.cu CPU_dp.cu -o main

exec_main_0:
	./main 1 30 100000 1

exec_main_1:
	./main 1 100 1000000000 1

exec_main_2:
	./main 1 100 10000000000 1