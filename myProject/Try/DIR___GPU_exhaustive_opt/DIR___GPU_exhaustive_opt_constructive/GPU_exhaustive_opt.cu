#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_DIM_X 128
#define MAX(a,b) (((a)>(b))?(a):(b))
#define N 10	//THIS IS PROBABLY THE MOST IMPORTANT THING: THE NUMBER OF VOLUMES IN THE PROBLEM (Must be set a priori, before COMPILING the code!)


//First, we need a very simple procedure to generate the bit string that will represent the current index
//of the exhaustive search.
//n: the total number of different items (copies included)
char* produce_initial_string(int n){
	char* res = (char*) malloc(n*sizeof(char));
	for(int i = 0; i < n; i++){
		res[i] = '0';
	}
	return res;
}

//Second, to sum binary strings together, we need a procedure thatn makes a bit string as long as another one
//Adapted from: https://www.geeksforgeeks.org/add-two-bit-strings/
//We always assume that the length of to_modify is less or equal than n
//Returns to_modify of the right length
__host__ __device__ char* extend_bit_string(int n, char* to_modify, int n_to_modify){	
	char* res = (char*) malloc(n * sizeof(char));
	int j = n_to_modify - 1;
	for(int i = n - 1; i >= n - n_to_modify; i--){
		res[i] = to_modify[j];
		j--;
	}
	for(int i = 0; i < n - n_to_modify; i++){
		res[i] = '0';
	}
	return res; 
}

//Third, we need the actual code that sums two bit strings of the same length.
//This assumes that the two strings have the same length n.
//The result is stored in the first string.
//Adapted from: https://www.geeksforgeeks.org/add-two-bit-strings/
__host__ __device__ void add_bit_strings(char** str1, char* str2, int n)
{
	//Initialize carry
	int carry = (int) '0'; 

	//Add all bits one by one
	for (int i = n - 1; i >= 0; i--)
	{
			int firstBit = (int) (*str1)[i];
			int secondBit = (int) str2[i];

			//Boolean expression for sum of 3 bits
			int sum = (firstBit ^ secondBit ^ carry);

			(*str1)[i] = (char) sum;

			// boolean expression for 3-bit addition
			carry = (firstBit & secondBit) |
								 (secondBit & carry) |
									(firstBit & carry);
	}

	// if overflow, then add a leading 1
	if (carry == 1)
	{
		printf("overflow!");
		//result = "1" + result;
	}
}

//Fourth, a function that translates an integer into its bit string representation
__host__ __device__ char* convert_to_binary(int n){
	int k;
	int l = (int) log2f(n) + 1;
	char* res = (char*) malloc(l * sizeof(char));
	int i = 0;
	for(int c = l - 1; c >= 0; c--){
		k = n >> c;
		if (k & 1){
			res[i] = '1';
		}else{
			res[i] = '0';
		}
		i++;
	}
	return res;
}

//Fifth, a function that checks whether two binary strings of the same length are the same or not
int equal_bit_strings(char* str1, char* str2, int n){
	for(int i = 0; i < n; i++){
		if(str1[i] != str2[i]){
			return 0;
		}
	}
	return 1;
	
}

//Sixth, a function that, given a binary string of n_volumes and the array of volumes, returns the
//sum of the volumes with entry 1 in the binary string
__host__ __device__ int value_of_solution(char* bit_string, int* volumes, int n){
	int sum = 0;
	for(int i = 0; i < n; i++){
		sum = sum + ((bit_string[i] - '0') * volumes[i]);
	}
	return sum;
}



//Function for the qsort to order in decreasing order
int comp_decr(const void * a, const void * b)
{
 return *(const int*)b - *(const int*)a;
}









//The kernel that applies the parallel algorithm
//The idea is:
//each thread, given the current global_index_counter and its thread+block index, determines its local index.
//Then, it uses that index (together with the volumes) to compute the solution of this node of the exponential tree.
__global__ void kernel_exhaustive(int* volumes, int n_volumes, int capacity, char* global_index_counter, int* partial_results){
	//First: compute the global grid index of this thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Second: convert it to its binary, extended representation.
	char* binary_idx = convert_to_binary(idx);
	char* extended_binary_idx = extend_bit_string(n_volumes, binary_idx, (int) log2f(idx) + 1);
	printf("HIUIIII, idx = %d\n", idx);
	free(binary_idx);
	binary_idx = extended_binary_idx;

	//Third: add this value to the global_index_counter passed as argument
	add_bit_strings(&global_index_counter, binary_idx, n_volumes);

	//Fourth: compute the current solution
	int sum = value_of_solution(global_index_counter, volumes, n_volumes);

	//Fifth: if the value of the current solution is legal (does not exceed capacity), we return it. Else,
	//We return 0. Since from all the values computed by kernel instances of this kind only the maximum
	//is returned as final result, this does not change the correctness of the algorithm.
	//This introduces (possibly, only in some cases) a bit of divergence, but it's just a line of code
	//(and I mean, it's an operation that before or after has to be performed).
	printf("sum = %d, capacity = %d\n", sum, capacity);
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
	if (blockDim.x>=128 && tid < 64) idata[tid] += MAX(idata[tid], idata[tid + 64]);
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
int subsetSumOptimization_exhaustive_GPU(int* volumes, int capacity, int n_volumes, int jump){
	
	//First: order the volumes in decreasing order
	qsort(volumes, n_volumes, sizeof(int), comp_decr);
	
	//Second: produce the initial bit string of length n_volumes
	char* global_index_counter = produce_initial_string(n_volumes);
	
	//Third: translate the jump value into its binary form
	char* jump_binary = convert_to_binary(jump);
	
	//Fourth: extend the jump binary representation to the length of the global index counter
	char* jump_binary_extended = extend_bit_string(n_volumes, jump_binary, (int) log2f(jump) + 1);
	free(jump_binary);
	jump_binary = jump_binary_extended;
	
	char* starting_index = (char*) malloc(n_volumes * sizeof(char));
	for(int i = 0; i < n_volumes; i++){
		starting_index[i] = global_index_counter[i];
	}

	dim3 block(BLOCK_DIM_X);
	dim3 grid((jump + block.x - 1) / block.x);
	dim3 grid8((grid.x + 8 - 1)/8);

	int* partial_results;
	cudaMalloc((int**) &partial_results, jump * sizeof(int));

	int partial_reductions_len = ((grid.x + 8 - 1)/8);
	int* partial_reductions_GPU;
	cudaMalloc((int**) &partial_reductions_GPU, partial_reductions_len * sizeof(int));

	int* partial_reductions_CPU = (int*) malloc(partial_reductions_len * sizeof(int));
	//setup phase ended

	int FINAL_RESULT = 0;

	do{
		printf("global_index_counter = ");
		for(int i = 0; i < n_volumes; i++){
			printf("%c", global_index_counter[i]);
		}
		printf("\n");

		//IF IL VALORE CORRENTE DELL'INDICE GLOBALE E' MINORE DELLA CAPACITA' ???
		if(value_of_solution(global_index_counter, volumes, n_volumes) <= capacity){
			printf("HI\n");
			//This is the main loop that does the important computations.
			//First: launch a kernel that will compute all the tentative solutions of index between
			//"global_index_counter" and "global_index_counter" + "jump"
			kernel_exhaustive<<<grid, block>>>(volumes, n_volumes, capacity, global_index_counter, partial_results);

			//Second: launch a second, reduction kernel, that will find, among these values, the one with
			//the highest value (lower than capacity)
			reduceCompleteUnrollWarps8<<<grid8, block>>>(partial_results, partial_reductions_GPU, grid.x * block.x);

			//Third: reduce one last time on the CPU
			cudaMemcpy(partial_reductions_CPU, partial_reductions_GPU, partial_reductions_len * sizeof(int), cudaMemcpyDeviceToHost);
			int new_solution = 0;
			for(int i = 0; i < partial_reductions_len; i++){
				new_solution += partial_reductions_CPU[i];
			}
			
			//Fourth: update the new best solution
			if(new_solution > FINAL_RESULT) FINAL_RESULT = new_solution;
		}

		//Fifth: update the index
		add_bit_strings(&global_index_counter, jump_binary, n_volumes);
		
	}while(!equal_bit_strings(global_index_counter, starting_index, n_volumes));
	
	return FINAL_RESULT;	//needs to return the actual result of the whole problem
	
}










int main(){
	int n = 10;
	int c = 10000;
	int j = 1024/8;
	int* volumes = (int*) malloc(n*sizeof(int));
	for(int i = 0; i < n; i++){
		volumes[i] = i*10;
	}
	int res = subsetSumOptimization_exhaustive_GPU(volumes, c, n, j);
	printf("result = %d", res);
	return 0;
}