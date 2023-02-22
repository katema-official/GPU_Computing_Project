#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "../../../common.h"
#include "../config.h"
#include "GPU_exhaustive_opt_common.h"



//The kernel that applies the parallel algorithm
//The idea is:
//each thread, given the current global_index_counter and its thread+block index, determines its local index.
//Then, it uses that index (together with the volumes) to compute the solution of this node of the exponential tree.
__global__ void kernel_exhaustive(int capacity, char global_index_counter[N], int* partial_results){
	//First: compute the global grid index of this thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Second: convert it to its binary representation.
	char binary_idx[N];
	convert_to_binary(binary_idx, idx);

	//Third: make a copy of the global index counter
	//Let's use unrolling also here. We propose an unrolling factor of 8
	int uroll_iterations = N/8;
	char global_index_counter_local[N];
	for(int i = 0; i < uroll_iterations*8; i+=8){
		global_index_counter_local[i] = global_index_counter[i];
		global_index_counter_local[i + 1] = global_index_counter[i + 1];
		global_index_counter_local[i + 2] = global_index_counter[i + 2];
		global_index_counter_local[i + 3] = global_index_counter[i + 3];
		global_index_counter_local[i + 4] = global_index_counter[i + 4];
		global_index_counter_local[i + 5] = global_index_counter[i + 5];
		global_index_counter_local[i + 6] = global_index_counter[i + 6];
		global_index_counter_local[i + 7] = global_index_counter[i + 7];
	}
	for(int i = 8*uroll_iterations; i < N; i++){
		global_index_counter_local[i] = global_index_counter[i];
	}

	//Third: add this value to the global_index_counter passed as argument
	add_bit_strings(global_index_counter_local, binary_idx);

	//Fourth: compute the current solution
	int sum = value_of_solution_device(global_index_counter_local);

	//Fifth: if the value of the current solution is legal (does not exceed capacity), we return it. Else,
	//We return 0. Since from all the values computed by kernel instances of this kind only the maximum
	//is returned as final result, this does not change the correctness of the algorithm.
	//This introduces (possibly, only in some cases) a bit of divergence, but it's just a line of code
	//(and I mean, it's an operation that before or after has to be performed).
	if(sum > capacity) sum = 0;

	//Fifth: store it in global memory
	partial_results[idx] = sum;


}

//reduction kernel (taken (and adapted a bit) from Professional CUDA C Programming)
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, int n){
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x * 8;
	// unrolling 8
	int a1 = 0;
	int a2 = 0;
	int a3 = 0;
	int a4 = 0;
	int b1 = 0;
	int b2 = 0;
	int b3 = 0;
	int b4 = 0;
	if (idx + 7*blockDim.x < n) {
		a1 = g_idata[idx];
		a2 = g_idata[idx + blockDim.x];
		a3 = g_idata[idx + 2 * blockDim.x];
		a4 = g_idata[idx + 3 * blockDim.x];
		b1 = g_idata[idx + 4 * blockDim.x];
		b2 = g_idata[idx + 5 * blockDim.x];
		b3 = g_idata[idx + 6 * blockDim.x];
		b4 = g_idata[idx + 7 * blockDim.x];
	}else{
		if(idx < n) a1 = g_idata[idx];
		if(idx + blockDim.x < n) a2 = g_idata[idx + blockDim.x];
		if(idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
		if(idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
		if(idx + 4 * blockDim.x < n) b1 = g_idata[idx + 4 * blockDim.x];
		if(idx + 5 * blockDim.x < n) b2 = g_idata[idx + 5 * blockDim.x];
		if(idx + 6 * blockDim.x < n) b3 = g_idata[idx + 6 * blockDim.x];
	}
	g_idata[idx] = MAX(a1, MAX(a2, MAX(a3, MAX(a4, MAX(b1, MAX(b2, MAX(b3, b4)))))));
	__syncthreads();
	// in-place reduction and complete unroll
	if (blockDim.x>=1024 && tid < 512) idata[tid] = MAX(idata[tid], idata[tid + 512]);
	__syncthreads();
	if (blockDim.x>=512 && tid < 256) idata[tid] = MAX(idata[tid], idata[tid + 256]);
	__syncthreads();
	if (blockDim.x>=256 && tid < 128) idata[tid] = MAX(idata[tid], idata[tid + 128]);
	__syncthreads();
	if (blockDim.x>=128 && tid < 64) idata[tid] = MAX(idata[tid], idata[tid + 64]);
	__syncthreads();
	// unrolling warp
	if (tid < 32) {
		volatile int *vsmem = idata;
		vsmem[tid] = MAX(vsmem[tid], vsmem[tid + 32]);
		vsmem[tid] = MAX(vsmem[tid], vsmem[tid + 16]);
		vsmem[tid] = MAX(vsmem[tid], vsmem[tid + 8]);
		vsmem[tid] = MAX(vsmem[tid], vsmem[tid + 4]);
		vsmem[tid] = MAX(vsmem[tid], vsmem[tid + 2]);
		vsmem[tid] = MAX(vsmem[tid], vsmem[tid + 1]);
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}











//the function to call externally. Arguments:
//volumes: the array of volumes, that can also have copies
//capacity: the target value
//n_volumes: the length of "volumes"
//jump: a value that must be a power of 2, and that decides how big is a batch of elements to be examined.
int subsetSumOptimization_exhaustive_GPU(int volumes[N], int capacity, int jump){
	
	//First: order the volumes in decreasing order
	qsort(volumes, N, sizeof(int), comp_decr);
	
	//Second: produce the initial bit string of length N (number of volumes)
	char global_index_counter[N];
	produce_initial_string(global_index_counter);
	
	//Third: translate the jump value into its binary form
	char jump_binary[N];
	convert_to_binary(jump_binary, jump);

	for(int i = 0; i < N; i++){
		printf("%c", jump_binary[i]);
	}
	printf("\n");

	//Fourth: get a copy of the starting index, so that we know when to stop (when we overflow and reach the initial value again)
	char starting_index[N];
	for(int i = 0; i < N; i++){
		starting_index[i] = global_index_counter[i];
	}


	//Now, we need to setup the CUDA environment.
	//1) Declare block and grid size for the kernels
	dim3 block(BLOCK_DIM_X);
	dim3 grid((jump + block.x - 1) / block.x);
	dim3 grid8((grid.x + 8 - 1)/8);
	printf("block.x = %d, grid.x = %d\n", block.x, grid.x);

	//2) allocate on the device enough memory to store the tentative results (of the problem) produced by each kernel (1)
	int* partial_results_GPU;
	cudaMalloc((int**) &partial_results_GPU, jump * sizeof(int));

	//3) allocate on the device enough memory to store the reduction results (given by the previous kernel) that will be produced by the reduction kernel
	int partial_reductions_len = grid8.x;
	int* partial_reductions_GPU;
	cudaMalloc((int**) &partial_reductions_GPU, partial_reductions_len * sizeof(int));

	//4) allocate on the host enough memory to store the results of the reduction kernel, so that they can be further reduced
	int* partial_reductions_CPU = (int*) malloc(partial_reductions_len * sizeof(int));

	
	//5) allocate on the device enough memory to store the current global index
	char* global_index_counter_GPU;
	cudaMalloc((char**) &global_index_counter_GPU, N * sizeof(char));

	//6) [allocate and copy on the device the volumes] NO! Use constant memory instead!
	init_volumes_constant(volumes);
	
	//cudaMalloc((int**) &volumes_GPU, N * sizeof(int));
	//cudaMemcpy(volumes_GPU, volumes, N * sizeof(int), cudaMemcpyHostToDevice);
	//setup phase ended

	int FINAL_RESULT = 0;

	do{
		//If the current value of the global_index_counter, translated in a solution, is less than the capacity 
		//(which means that this cunck of tentative solutions needs to be explored, because at least one solution is admissible)
		if(value_of_solution(global_index_counter, volumes) <= capacity){
			//This is the main loop that does the important computations.

			/*for(int i = 0; i < N; i++){
				printf("%c", global_index_counter[i]);
			}
			printf("\n");*/

			//First: copy the current global_index_counter value to the device
			cudaMemcpy(global_index_counter_GPU, global_index_counter, N * sizeof(char), cudaMemcpyHostToDevice);

			//Second: launch a kernel that will compute all the tentative solutions of index between
			//"global_index_counter" and "global_index_counter" + "jump"
			kernel_exhaustive<<<grid, block>>>(capacity, global_index_counter_GPU, partial_results_GPU);

			//Third: launch a second, reduction kernel, that will find, among these values, the one with
			//the highest value (lower than capacity)
			reduceCompleteUnrollWarps8<<<grid8, block>>>(partial_results_GPU, partial_reductions_GPU, jump);

			//Fourth: reduce one last time on the CPU
			cudaMemcpy(partial_reductions_CPU, partial_reductions_GPU, partial_reductions_len * sizeof(int), cudaMemcpyDeviceToHost);
			int new_solution = 0;
			for(int i = 0; i < partial_reductions_len; i++){
				new_solution = MAX(new_solution, partial_reductions_CPU[i]);
			}
			
			//Fifth: update the new best solution
			if(new_solution > FINAL_RESULT) FINAL_RESULT = new_solution;

			//If we are very lucky...
			//if(FINAL_RESULT == capacity) break;
		}

		//Sixth: update the index
		add_bit_strings(global_index_counter, jump_binary);
		
	}while(!equal_bit_strings(global_index_counter, starting_index));
	
	return FINAL_RESULT;	//needs to return the actual result of the whole problem
	
}


