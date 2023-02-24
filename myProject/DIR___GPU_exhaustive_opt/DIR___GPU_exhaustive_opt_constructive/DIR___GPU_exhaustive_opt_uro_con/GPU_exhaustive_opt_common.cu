#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "GPU_exhaustive_opt_common.h"
#include "../config.h"


__constant__ int volumes_constant[N];



//First, we need a very simple procedure to generate the bit string that will represent the current index
//of the exhaustive search.
//n: the total number of different items (copies included)
void produce_initial_string(char res[N]){
	for(int i = 0; i < N; i++){
		res[i] = '0';
	}
}

//Second, we need the code that sums two bit strings of the same length.
//This assumes that the two strings have the same length N.
//The result is stored in the first string.
//Adapted from: https://www.geeksforgeeks.org/add-two-bit-strings/
__host__ __device__ void add_bit_strings(char str1[N], char str2[N]){
	//Initialize carry
	int carry = (int) '0'; 

	//Add all bits one by one
	for (int i = N - 1; i >= 0; i--)
	{
		int firstBit = (int) str1[i];
		int secondBit = (int) str2[i];

		//Boolean expression for sum of 3 bits
		int sum = (firstBit ^ secondBit ^ carry);

		str1[i] = (char) sum;

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

//Third, a function that translates an integer into its bit string representation.
//The char array in which the binary representation will be conserved is passed as argument
__host__ __device__ void convert_to_binary(char res[N], int number){
	int k;
	int l = (int) log2f(number) + 1;
	int i = N - l;
	for(int c = l - 1; c >= 0; c--){
		k = number >> c;
		res[i] = ((char) k&1) + '0';	//this avoids warp divergence
		i++;
	}
	for(int c = 0; c < N - l; c++){
		res[c] = '0';
	}

}

//Fourth, a function that checks whether two binary strings of the same length are the same or not
int equal_bit_strings(char* str1, char* str2){
	for(int i = 0; i < N; i++){
		if(str1[i] != str2[i]){
			return 0;
		}
	}
	return 1;
	
}





//Fifth, a function that, given a binary string of n_volumes and the array of volumes, returns the
//sum of the volumes with entry 1 in the binary string
//We propose, this time, weo variants of this function:
//-one for the host, as before
//-one for the device, with unrolling AND constant memory
int value_of_solution(char bit_string[N], int volumes[N]){
	int sum = 0;
	for(int i = 0; i < N; i++){
		sum = sum + ((bit_string[i] - '0') * volumes[i]);
	}
	return sum;
}

//the chosen unrolling is 8
__device__ int value_of_solution_device(char bit_string[N]){
	int uroll_iterations = N/8;
	int a[8];
	int sum = 0;

	for(int i = 0; i < uroll_iterations*8; i += 8){
		a[0] = ((bit_string[i + 0] - '0') * volumes_constant[i + 0]);
		a[1] = ((bit_string[i + 1] - '0') * volumes_constant[i + 1]);
		a[2] = ((bit_string[i + 2] - '0') * volumes_constant[i + 2]);
		a[3] = ((bit_string[i + 3] - '0') * volumes_constant[i + 3]);
		a[4] = ((bit_string[i + 4] - '0') * volumes_constant[i + 4]);
		a[5] = ((bit_string[i + 5] - '0') * volumes_constant[i + 5]);
		a[6] = ((bit_string[i + 6] - '0') * volumes_constant[i + 6]);
		a[7] = ((bit_string[i + 7] - '0') * volumes_constant[i + 7]);
		sum = sum + a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
	}
	for(int i = uroll_iterations*8; i < N; i++){
		sum = sum + ((bit_string[i] - '0') * volumes_constant[i]);
	}
	return sum;
}










//Function for the qsort to order in decreasing order
int comp_decr(const void * a, const void * b)
{
 return *(const int*)b - *(const int*)a;
}





//function to initialize the constant memory

void init_volumes_constant(int volumes[N]){
	cudaMemcpyToSymbol(volumes_constant, volumes, N * sizeof(int), 0, cudaMemcpyHostToDevice);
}