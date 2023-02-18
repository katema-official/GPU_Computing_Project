#ifndef GPU_dp_uroll
#define GPU_dp_uroll

__global__ void kernel_v2_a_uroll8(unsigned char* row_0, int capacity);
__global__ void kernel_v2_b_uroll8(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx);
unsigned char DP_v2_uroll8_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block);

__global__ void kernel_v2_a_uroll8_v2(unsigned char* row_0, int capacity);
__global__ void kernel_v2_b_uroll8_v2(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx);
unsigned char DP_v2_uroll8_GPU_v2(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block);

#endif