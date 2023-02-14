compile_main1: main1.cu CPU_dp.cu CPU_exhaustive_dec.cu CPU_exhaustive_opt.cu GPU_dp_basic.cu GPU_dp_smem.cu GPU_dp_uroll.cu GPU_dp_uroll8_smem.cu
	nvcc -arch=sm_37 main1.cu CPU_dp.cu CPU_exhaustive_dec.cu CPU_exhaustive_opt.cu GPU_dp_basic.cu GPU_dp_smem.cu GPU_dp_uroll.cu GPU_dp_uroll8_smem.cu -o main1

exec_main1_1:
	./main1 1 100 1000000000 1