#ifndef GPU_EXHAUSTIVE_OPT_COMMON
#define GPU_EXHAUSTIVE_OPT_COMMON

#define MAX(a,b) (((a)>(b))?(a):(b))
#define N 50	//THIS IS PROBABLY THE MOST IMPORTANT THING: THE NUMBER OF VOLUMES IN THE PROBLEM (Must be set a priori, before COMPILING the code!)


void produce_initial_string(char res[N]);
__host__ __device__ void add_bit_strings(char str1[N], char str2[N]);
__host__ __device__ void convert_to_binary(char res[N], int number);
int equal_bit_strings(char* str1, char* str2);
__host__ __device__ int value_of_solution(char bit_string[N], int volumes[N]);
int comp_decr(const void * a, const void * b);

#endif