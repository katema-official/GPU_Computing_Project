#include "common.h"
#include <stdio.h>

//v2 smem: as v2, but we bring the necessary data in the shared memory, then we read the data from there

__global__ void kernel_v2_a_smem(unsigned char* row_0, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx == 0){
    row_0[idx] = TRUE;
  }else{
    if(idx <= capacity){
      row_0[idx] = FALSE;
    }
  }
}

__global__ void kernel_v2_b_smem(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //use shared memory to write and read data. We change the behaviour depending on the position of the block:
  //1) all the threads have an idx < v
  //2) some threads have an idx < v, but some do not
  //3) all the threads have an idx >= v

  int situation = 0;

  if(blockIdx.x * blockDim.x + blockDim.x - 1 < v){
    //section 1)
    situation = 1;
  }else{
    if(blockIdx.x * blockDim.x < v && blockIdx.x * blockDim.x + blockDim.x - 1 >= v){
      //section 2)
      situation = 2;
    }else{
      //section 3)
      situation = 3;
    }
  }

  __shared__ unsigned char smem[BLOCK_DIM_X * 2];

  unsigned char a;
  unsigned char b;

  int warp_id;

  if(old_row_idx == 0){

    //compute row_1 from row_0

    switch(situation){
      case 1:
        smem[threadIdx.x] = row_0[idx];
        row_1[idx] = smem[threadIdx.x];
      break;
      case 2:
        warp_id = threadIdx.x / 32;
        if(idx <= capacity){
          if(idx < v){
            smem[warp_id * 32 + threadIdx.x + 32] = row_0[idx];
            a = FALSE;
            b = smem[warp_id * 32 + threadIdx.x + 32];
          }else{
            smem[warp_id * 32 + threadIdx.x] = row_0[idx - v];
            smem[warp_id * 32 + threadIdx.x + 32] = row_0[idx];
            a = smem[warp_id * 32 + threadIdx.x];
            b = smem[warp_id * 32 + threadIdx.x + 32];
          }
          row_1[idx] = b || a;
        }
      break;
      case 3:
        warp_id = threadIdx.x / 32;
        if(idx <= capacity){
          smem[warp_id * 32 + threadIdx.x] = row_0[idx - v];
          smem[warp_id * 32 + threadIdx.x + 32] = row_0[idx];
          a = smem[warp_id * 32 + threadIdx.x];
          b = smem[warp_id * 32 + threadIdx.x + 32];
          row_1[idx] = b || a;
        }
      break;
    }

  }else{

    //compute row_0 from row_1

    switch(situation){
      case 1:
        smem[threadIdx.x] = row_1[idx];
        row_0[idx] = smem[threadIdx.x];
      break;
      case 2:
        warp_id = threadIdx.x / 32;
        if(idx <= capacity){
          if(idx < v){
            smem[warp_id * 32 + threadIdx.x + 32] = row_1[idx];
            a = FALSE;
            b = smem[warp_id * 32 + threadIdx.x + 32];
          }else{
            smem[warp_id * 32 + threadIdx.x] = row_1[idx - v];
            smem[warp_id * 32 + threadIdx.x + 32] = row_1[idx];
            a = smem[warp_id * 32 + threadIdx.x];
            b = smem[warp_id * 32 + threadIdx.x + 32];
          }
          row_0[idx] = b || a;
        }
      break;
      case 3:
        warp_id = threadIdx.x / 32;
        if(idx <= capacity){
          smem[warp_id * 32 + threadIdx.x] = row_1[idx - v];
          smem[warp_id * 32 + threadIdx.x + 32] = row_1[idx];
          a = smem[warp_id * 32 + threadIdx.x];
          b = smem[warp_id * 32 + threadIdx.x + 32];
          row_0[idx] = b || a;
        }
      break;
    }

  }
  
}

//Function that uses the two kernels above to solve the SubsetSumDecision problem

unsigned char DP_v2_smem_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block){
  unsigned char result = FALSE;
  
  //first step: create the initial row
  kernel_v2_a_smem<<<grid, block>>>(old_row_d, capacity);
  CHECK(cudaDeviceSynchronize());

  //second step: compute the new row from the old one
  int old_row_idx = 0;
  for(int r = 0; r < n_items; r++){
    kernel_v2_b_smem<<<grid, block>>>(volumes[r], old_row_d, new_row_d, capacity, old_row_idx);
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