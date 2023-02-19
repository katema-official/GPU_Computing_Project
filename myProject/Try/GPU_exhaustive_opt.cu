#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


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
char* extend_bit_string(int n, char* to_modify, int n_to_modify){	
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
void add_bit_strings(char* str1, char* str2, int n)
{
	//Initialize carry
	int carry = (int) '0'; 

	//Add all bits one by one
	for (int i = n - 1; i >= 0; i--)
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

//Fourth, a function that translates an integer into its bit string representation
char* convert_to_binary(int n){
	int k;
	int l = (int) log2(n) + 1;
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
	printf("\n");
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





//Function for the qsort to order in decreasing order
int comp_decr(const void * a, const void * b)
{
 return *(const int*)b - *(const int*)a;
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
	char* jump_binary_extended = extend_bit_string(n_volumes, jump_binary, (int) log2(jump) + 1);
	free(jump_binary);
	jump_binary = jump_binary_extended;
	
	
	while(equal_bit_strings(global_index_counter, jump_binary, n_volumes)){
		printf("global_index_counter = ");
		for(int i = 0; i < n_volumes; i++){
			printf("%c", global_index_counter[i]);
		}
		printf("\n");
		
		add_bit_strings(global_index_counter, jump_binary, n_volumes);
	}
	
	return 1;
	
}

int main(){
	int n = 20;
	int c = 10000;
	int j = 1024;
	int* volumes = (int*) malloc(n*sizeof(int));
	for(int i = 0; i < 20; i++){
		volumes[i] = i*10;
	}
	subsetSumOptimization_exhaustive_GPU(volumes, c, n, j);
	return 0;
}