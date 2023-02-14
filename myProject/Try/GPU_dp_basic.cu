#include "common.h"
#include <stdio.h>

//v1: Compute on the device one row, than bring it back to the host.
//then copy it again on the device and use it to compute the new row.
//-Inefficient because of multiple copies between device and host

__global__ void kernel_v1_a(unsigned char* res_row, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx == 0){
    res_row[idx] = TRUE;
  }else{
    if(idx <= capacity){
        res_row[idx] = FALSE;
    }
  }
}

__global__ void kernel_v1_b(int v, unsigned char* input_row, unsigned char* output_row, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < v){
    output_row[idx] = input_row[idx];
  }else{
    if(idx <= capacity) output_row[idx] = input_row[idx] || input_row[idx - v];
  }
}

//Function that uses the two kernels above to solve the SubsetSumDecision problem

unsigned char DP_v1_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block){
  unsigned char result = FALSE;

  //first step: create the initial row
  kernel_v1_a<<<grid, block>>>(old_row_d, capacity);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(row_h, old_row_d, (capacity + 1)*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  if(DEBUG_1) printf("temporary result: %d\n", row_h[capacity]);

  //second step: create all the subsequent rows
  for(int r = 0; r < n_items; r++){
    CHECK(cudaMemcpy(old_row_d, row_h, (capacity + 1)*sizeof(unsigned char), cudaMemcpyHostToDevice));
    kernel_v1_b<<<grid, block>>>(volumes[r], old_row_d, new_row_d, capacity);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(row_h, new_row_d, (capacity + 1)*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    if(DEBUG_1) printf("temporary result: %d\n", row_h[capacity]);
    if(row_h[capacity] == TRUE){
      result = TRUE;
      break;
    }
  }

  return result;

}






//v2: do everything in one kernel, minimizing the copies in global memory
//and doing only the strictly necessary data transfers between device and host

__global__ void kernel_v2_a(unsigned char* row_0, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx == 0){
    row_0[idx] = TRUE;
  }else{
    if(idx <= capacity){
      row_0[idx] = FALSE;
    }
  }
}

__global__ void kernel_v2_b(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //now, take the last row that has been computed, and use it to compute the new
  //one, that will be the opposite row. This avoids us to copy data from row_1
  //to row_0 for o(vols) times.
  
  if(old_row_idx == 0){

    //compute row_1 from row_0
    if(idx < v){
      row_1[idx] = row_0[idx];
    }else{
      if(idx <= capacity) row_1[idx] = row_0[idx] || row_0[idx - v];
    }

  }else{

    //compute row_0 from row_1
    if(idx < v){
      row_0[idx] = row_1[idx];
    }else{
      if(idx <= capacity) row_0[idx] = row_1[idx] || row_1[idx - v];
    }
  }
}

//Function that uses the two kernels above to solve the SubsetSumDecision problem

unsigned char DP_v2_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block){
  unsigned char result = FALSE;
  
  //first step: create the initial row
  kernel_v2_a<<<grid, block>>>(old_row_d, capacity);
  CHECK(cudaDeviceSynchronize());

  //second step: compute the new row from the old one
  int old_row_idx = 0;
  for(int r = 0; r < n_items; r++){
    kernel_v2_b<<<grid, block>>>(volumes[r], old_row_d, new_row_d, capacity, old_row_idx);
    CHECK(cudaDeviceSynchronize());
    old_row_idx = 1 - old_row_idx;  //switch the row
    
    if(old_row_idx == 0) cudaMemcpy(&result, &old_row_d[capacity], sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if(old_row_idx == 1) cudaMemcpy(&result, &new_row_d[capacity], sizeof(unsigned char), cudaMemcpyDeviceToHost);

    if(result == TRUE){
      break;
    }
  }

  return result;
}