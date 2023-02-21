#ifndef GPU_dp_basic
#define GPU_dp_basic

__global__ void kernel_v1_a(unsigned char* res_row, int capacity);
__global__ void kernel_v1_b(int v, unsigned char* input_row, unsigned char* output_row, int capacity);
unsigned char DP_v1_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block);

__global__ void kernel_v2_a(unsigned char* row_0, int capacity);
__global__ void kernel_v2_b(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx);
unsigned char DP_v2_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block);

#endif
