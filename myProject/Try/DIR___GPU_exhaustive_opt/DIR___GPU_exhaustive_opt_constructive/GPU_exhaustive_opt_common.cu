#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

//#include "../common.h"
#include "GPU_exhaustive_opt_common.h"


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
		if (k & 1){
			res[i] = '1';
		}else{
			res[i] = '0';
		}
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
__host__ __device__ int value_of_solution(char bit_string[N], int volumes[N]){
	int sum = 0;
	for(int i = 0; i < N; i++){
		sum = sum + ((bit_string[i] - '0') * volumes[i]);
	}
	return sum;
}



//Function for the qsort to order in decreasing order
int comp_decr(const void * a, const void * b)
{
 return *(const int*)b - *(const int*)a;
}