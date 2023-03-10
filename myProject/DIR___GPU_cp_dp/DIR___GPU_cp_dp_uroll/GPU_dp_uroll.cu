#include "../../common.h"
#include <stdio.h>

//v2 uroll8: as v2, but we give more work to each single thread with the unrolling technique

__global__ void kernel_v2_a_uroll8(unsigned char* row_0, int capacity){
  int idx_initialData = blockIdx.x * blockDim.x * 8 + threadIdx.x * 8;
  int idx = idx_initialData;

  if(idx == 0){
    row_0[idx] = TRUE;
    row_0[idx + 1] = FALSE;
    row_0[idx + 2] = FALSE;
    row_0[idx + 3] = FALSE;
    row_0[idx + 4] = FALSE;
    row_0[idx + 5] = FALSE;
    row_0[idx + 6] = FALSE;
    row_0[idx + 7] = FALSE;
  }else{
    if(idx + 7 <= capacity){
      row_0[idx] = FALSE;
      row_0[idx + 1] = FALSE;
      row_0[idx + 2] = FALSE;
      row_0[idx + 3] = FALSE;
      row_0[idx + 4] = FALSE;
      row_0[idx + 5] = FALSE;
      row_0[idx + 6] = FALSE;
      row_0[idx + 7] = FALSE;
    }else{
      //the last thread
      if(idx <= capacity)     row_0[idx] = FALSE;
      if(idx + 1 <= capacity) row_0[idx + 1] = FALSE;
      if(idx + 2 <= capacity) row_0[idx + 2] = FALSE;
      if(idx + 3 <= capacity) row_0[idx + 3] = FALSE;
      if(idx + 4 <= capacity) row_0[idx + 4] = FALSE;
      if(idx + 5 <= capacity) row_0[idx + 5] = FALSE;
      if(idx + 6 <= capacity) row_0[idx + 6] = FALSE;
    }
  }
}

__global__ void kernel_v2_b_uroll8(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx){
  int idx_initialData = blockIdx.x * blockDim.x * 8 + threadIdx.x * 8;
  int idx = idx_initialData;

  int situation = 0;

  if(idx + 7 < v){
    //all the 8 contiguous elements have index less than the volume of the current object
    situation = 1;
  }

  if(idx < v && idx + 7 >= v){
    //in the 8 contiguous elements, the first part (of unknown length) has indexes less than v.
    //the second part (of unknown length once again) has indexes greater than v.
    situation = 2;
  }

  if(idx >= v){
    //all the 8 contiguous elements have indexes greater than v
    situation = 3;
  }
  
  if(old_row_idx == 0){

    //compute row_1 from row_0

    if(situation == 1){
      row_1[idx] = row_0[idx];
      row_1[idx + 1] = row_0[idx + 1];
      row_1[idx + 2] = row_0[idx + 2];
      row_1[idx + 3] = row_0[idx + 3];
      row_1[idx + 4] = row_0[idx + 4];
      row_1[idx + 5] = row_0[idx + 5];
      row_1[idx + 6] = row_0[idx + 6];
      row_1[idx + 7] = row_0[idx + 7];
    }else{
      if(situation == 2){
          
          if(idx < v){
            row_1[idx] = row_0[idx]; 
          }else{
            if(idx <= capacity) row_1[idx] = row_0[idx] || row_0[idx - v];
          } 

          if(idx + 1 < v){
            row_1[idx + 1] = row_0[idx + 1]; 
          }else{
            if(idx + 1 <= capacity) row_1[idx + 1] = row_0[idx + 1] || row_0[idx + 1 - v];
          } 

          if(idx + 2 < v){
            row_1[idx + 2] = row_0[idx + 2]; 
          }else{
            if(idx + 2 <= capacity) row_1[idx + 2] = row_0[idx + 2] || row_0[idx + 2 - v];
          } 

          if(idx + 3 < v){
            row_1[idx + 3] = row_0[idx + 3]; 
          }else{
            if(idx + 3 <= capacity) row_1[idx + 3] = row_0[idx + 3] || row_0[idx + 3 - v];
          } 

          if(idx + 4 < v){
            row_1[idx + 4] = row_0[idx + 4]; 
          }else{
            if(idx + 4 <= capacity) row_1[idx + 4] = row_0[idx + 4] || row_0[idx + 4 - v];
          } 

          if(idx + 5 < v){
            row_1[idx + 5] = row_0[idx + 5]; 
          }else{
            if(idx + 5 <= capacity) row_1[idx + 5] = row_0[idx + 5] || row_0[idx + 5 - v];
          } 

          if(idx + 6 < v){
            row_1[idx + 6] = row_0[idx + 6]; 
          }else{
            if(idx + 6 <= capacity) row_1[idx + 6] = row_0[idx + 6] || row_0[idx + 6 - v];
          } 

          if(idx + 7 < v){
            row_1[idx + 7] = row_0[idx + 7]; 
          }else{
            if(idx + 7 <= capacity) row_1[idx + 7] = row_0[idx + 7] || row_0[idx + 7 - v];
          } 
        
      }else{
        //situation 3, because the situation is either 1, 2 or 3.
        if(idx + 7 <= capacity){
          row_1[idx] = row_0[idx] || row_0[idx - v];
          row_1[idx + 1] = row_0[idx + 1] || row_0[idx + 1 - v];
          row_1[idx + 2] = row_0[idx + 2] || row_0[idx + 2 - v];
          row_1[idx + 3] = row_0[idx + 3] || row_0[idx + 3 - v];
          row_1[idx + 4] = row_0[idx + 4] || row_0[idx + 4 - v];
          row_1[idx + 5] = row_0[idx + 5] || row_0[idx + 5 - v];
          row_1[idx + 6] = row_0[idx + 6] || row_0[idx + 6 - v];
          row_1[idx + 7] = row_0[idx + 7] || row_0[idx + 7 - v];
        }else{
          if(idx <= capacity)     row_1[idx] = row_0[idx] || row_0[idx - v];
          if(idx + 1 <= capacity) row_1[idx + 1] = row_0[idx + 1] || row_0[idx + 1 - v];
          if(idx + 2 <= capacity) row_1[idx + 2] = row_0[idx + 2] || row_0[idx + 2 - v];
          if(idx + 3 <= capacity) row_1[idx + 3] = row_0[idx + 3] || row_0[idx + 3 - v];
          if(idx + 4 <= capacity) row_1[idx + 4] = row_0[idx + 4] || row_0[idx + 4 - v];
          if(idx + 5 <= capacity) row_1[idx + 5] = row_0[idx + 5] || row_0[idx + 5 - v];
          if(idx + 6 <= capacity) row_1[idx + 6] = row_0[idx + 6] || row_0[idx + 6 - v];
        }
      }
    }

  }else{

    //compute row_0 from row_1

    if(situation == 1){
      row_0[idx] = row_1[idx];
      row_0[idx + 1] = row_1[idx + 1];
      row_0[idx + 2] = row_1[idx + 2];
      row_0[idx + 3] = row_1[idx + 3];
      row_0[idx + 4] = row_1[idx + 4];
      row_0[idx + 5] = row_1[idx + 5];
      row_0[idx + 6] = row_1[idx + 6];
      row_0[idx + 7] = row_1[idx + 7];
    }else{
      if(situation == 2){
          
          if(idx < v){
            row_0[idx] = row_1[idx]; 
          }else{
            if(idx <= capacity) row_0[idx] = row_1[idx] || row_1[idx - v];
          } 

          if(idx + 1 < v){
            row_0[idx + 1] = row_1[idx + 1]; 
          }else{
            if(idx + 1 <= capacity) row_0[idx + 1] = row_1[idx + 1] || row_1[idx + 1 - v];
          } 

          if(idx + 2 < v){
            row_0[idx + 2] = row_1[idx + 2]; 
          }else{
            if(idx + 2 <= capacity) row_0[idx + 2] = row_1[idx + 2] || row_1[idx + 2 - v];
          } 

          if(idx + 3 < v){
            row_0[idx + 3] = row_1[idx + 3]; 
          }else{
            if(idx + 3 <= capacity) row_0[idx + 3] = row_1[idx + 3] || row_1[idx + 3 - v];
          } 

          if(idx + 4 < v){
            row_0[idx + 4] = row_1[idx + 4]; 
          }else{
            if(idx + 4 <= capacity) row_0[idx + 4] = row_1[idx + 4] || row_1[idx + 4 - v];
          } 

          if(idx + 5 < v){
            row_0[idx + 5] = row_1[idx + 5]; 
          }else{
            if(idx + 5 <= capacity) row_0[idx + 5] = row_1[idx + 5] || row_1[idx + 5 - v];
          } 

          if(idx + 6 < v){
            row_0[idx + 6] = row_1[idx + 6]; 
          }else{
            if(idx + 6 <= capacity) row_0[idx + 6] = row_1[idx + 6] || row_1[idx + 6 - v];
          } 

          if(idx + 7 < v){
            row_0[idx + 7] = row_1[idx + 7]; 
          }else{
            if(idx + 7 <= capacity) row_0[idx + 7] = row_1[idx + 7] || row_1[idx + 7 - v];
          } 
        
      }else{
        //situation 3, because the situation is either 1, 2 or 3.
        if(idx + 7 <= capacity){
          row_0[idx] = row_1[idx] || row_1[idx - v];
          row_0[idx + 1] = row_1[idx + 1] || row_1[idx + 1 - v];
          row_0[idx + 2] = row_1[idx + 2] || row_1[idx + 2 - v];
          row_0[idx + 3] = row_1[idx + 3] || row_1[idx + 3 - v];
          row_0[idx + 4] = row_1[idx + 4] || row_1[idx + 4 - v];
          row_0[idx + 5] = row_1[idx + 5] || row_1[idx + 5 - v];
          row_0[idx + 6] = row_1[idx + 6] || row_1[idx + 6 - v];
          row_0[idx + 7] = row_1[idx + 7] || row_1[idx + 7 - v];
        }else{
          if(idx <= capacity)     row_0[idx] = row_1[idx] || row_1[idx - v];
          if(idx + 1 <= capacity) row_0[idx + 1] = row_1[idx + 1] || row_1[idx + 1 - v];
          if(idx + 2 <= capacity) row_0[idx + 2] = row_1[idx + 2] || row_1[idx + 2 - v];
          if(idx + 3 <= capacity) row_0[idx + 3] = row_1[idx + 3] || row_1[idx + 3 - v];
          if(idx + 4 <= capacity) row_0[idx + 4] = row_1[idx + 4] || row_1[idx + 4 - v];
          if(idx + 5 <= capacity) row_0[idx + 5] = row_1[idx + 5] || row_1[idx + 5 - v];
          if(idx + 6 <= capacity) row_0[idx + 6] = row_1[idx + 6] || row_1[idx + 6 - v];
        }
      }
    }

  }
}

//Function that uses the two kernels above to solve the SubsetSumDecision problem

unsigned char DP_v2_uroll8_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block){
  unsigned char result = FALSE;
  
  //first step: create the initial row
  kernel_v2_a_uroll8<<<grid, block>>>(old_row_d, capacity);
  CHECK(cudaDeviceSynchronize());

  //second step: compute the new row from the old one
  int old_row_idx = 0;
  for(int r = 0; r < n_items; r++){
    kernel_v2_b_uroll8<<<grid, block>>>(volumes[r], old_row_d, new_row_d, capacity, old_row_idx);
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





//v2 uroll16: as v2, but we give more work to each single thread with the unrolling technique

__global__ void kernel_v2_a_uroll16(unsigned char* row_0, int capacity){
  int idx_initialData = blockIdx.x * blockDim.x * 16 + threadIdx.x * 16;
  int idx = idx_initialData;

  if(idx == 0){
    row_0[idx] = TRUE;
    row_0[idx + 1] = FALSE;
    row_0[idx + 2] = FALSE;
    row_0[idx + 3] = FALSE;
    row_0[idx + 4] = FALSE;
    row_0[idx + 5] = FALSE;
    row_0[idx + 6] = FALSE;
    row_0[idx + 7] = FALSE;
    row_0[idx + 8] = FALSE;
    row_0[idx + 9] = FALSE;
    row_0[idx + 10] = FALSE;
    row_0[idx + 11] = FALSE;
    row_0[idx + 12] = FALSE;
    row_0[idx + 13] = FALSE;
    row_0[idx + 14] = FALSE;
    row_0[idx + 15] = FALSE;

  }else{
    if(idx + 15 <= capacity){
      row_0[idx] = FALSE;
      row_0[idx + 1] = FALSE;
      row_0[idx + 2] = FALSE;
      row_0[idx + 3] = FALSE;
      row_0[idx + 4] = FALSE;
      row_0[idx + 5] = FALSE;
      row_0[idx + 6] = FALSE;
      row_0[idx + 7] = FALSE;
      row_0[idx + 8] = FALSE;
      row_0[idx + 9] = FALSE;
      row_0[idx + 10] = FALSE;
      row_0[idx + 11] = FALSE;
      row_0[idx + 12] = FALSE;
      row_0[idx + 13] = FALSE;
      row_0[idx + 14] = FALSE;
      row_0[idx + 15] = FALSE;
    }else{
      //the last thread
      if(idx <= capacity)     row_0[idx] = FALSE;
      if(idx + 1 <= capacity) row_0[idx + 1] = FALSE;
      if(idx + 2 <= capacity) row_0[idx + 2] = FALSE;
      if(idx + 3 <= capacity) row_0[idx + 3] = FALSE;
      if(idx + 4 <= capacity) row_0[idx + 4] = FALSE;
      if(idx + 5 <= capacity) row_0[idx + 5] = FALSE;
      if(idx + 6 <= capacity) row_0[idx + 6] = FALSE;
      if(idx + 7 <= capacity) row_0[idx + 7] = FALSE;
      if(idx + 8 <= capacity) row_0[idx + 8] = FALSE;
      if(idx + 9 <= capacity) row_0[idx + 9] = FALSE;
      if(idx + 10 <= capacity) row_0[idx + 10] = FALSE;
      if(idx + 11 <= capacity) row_0[idx + 11] = FALSE;
      if(idx + 12 <= capacity) row_0[idx + 12] = FALSE;
      if(idx + 13 <= capacity) row_0[idx + 13] = FALSE;
      if(idx + 14 <= capacity) row_0[idx + 14] = FALSE;
    }
  }
}

__global__ void kernel_v2_b_uroll16(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx){
  int idx_initialData = blockIdx.x * blockDim.x * 16 + threadIdx.x * 16;
  int idx = idx_initialData;
  
  int situation = 0;

  if(idx + 16 < v){
    //all the 16 contiguous elements have index less than the volume of the current object
    situation = 1;
  }

  if(idx < v && idx + 16 >= v){
    //in the 16 contiguous elements, the first part (of unknown length) has indexes less than v.
    //the second part (of unknown length once again) has indexes greater than v.
    situation = 2;
  }

  if(idx >= v){
    //all the 16 contiguous elements have indexes greater than v
    situation = 3;
  }
  
  if(old_row_idx == 0){

    //compute row_1 from row_0

    if(situation == 1){
      row_1[idx] = row_0[idx];
      row_1[idx + 1] = row_0[idx + 1];
      row_1[idx + 2] = row_0[idx + 2];
      row_1[idx + 3] = row_0[idx + 3];
      row_1[idx + 4] = row_0[idx + 4];
      row_1[idx + 5] = row_0[idx + 5];
      row_1[idx + 6] = row_0[idx + 6];
      row_1[idx + 7] = row_0[idx + 7];
      row_1[idx + 8] = row_0[idx + 8];
      row_1[idx + 9] = row_0[idx + 9];
      row_1[idx + 10] = row_0[idx + 10];
      row_1[idx + 11] = row_0[idx + 11];
      row_1[idx + 12] = row_0[idx + 12];
      row_1[idx + 13] = row_0[idx + 13];
      row_1[idx + 14] = row_0[idx + 14];
      row_1[idx + 15] = row_0[idx + 15];
    }else{
      if(situation == 2){
          
          if(idx < v){
            row_1[idx] = row_0[idx]; 
          }else{
            if(idx <= capacity) row_1[idx] = row_0[idx] || row_0[idx - v];
          } 

          if(idx + 1 < v){
            row_1[idx + 1] = row_0[idx + 1]; 
          }else{
            if(idx + 1 <= capacity) row_1[idx + 1] = row_0[idx + 1] || row_0[idx + 1 - v];
          } 

          if(idx + 2 < v){
            row_1[idx + 2] = row_0[idx + 2]; 
          }else{
            if(idx + 2 <= capacity) row_1[idx + 2] = row_0[idx + 2] || row_0[idx + 2 - v];
          } 

          if(idx + 3 < v){
            row_1[idx + 3] = row_0[idx + 3]; 
          }else{
            if(idx + 3 <= capacity) row_1[idx + 3] = row_0[idx + 3] || row_0[idx + 3 - v];
          } 

          if(idx + 4 < v){
            row_1[idx + 4] = row_0[idx + 4]; 
          }else{
            if(idx + 4 <= capacity) row_1[idx + 4] = row_0[idx + 4] || row_0[idx + 4 - v];
          } 

          if(idx + 5 < v){
            row_1[idx + 5] = row_0[idx + 5]; 
          }else{
            if(idx + 5 <= capacity) row_1[idx + 5] = row_0[idx + 5] || row_0[idx + 5 - v];
          } 

          if(idx + 6 < v){
            row_1[idx + 6] = row_0[idx + 6]; 
          }else{
            if(idx + 6 <= capacity) row_1[idx + 6] = row_0[idx + 6] || row_0[idx + 6 - v];
          } 

          if(idx + 7 < v){
            row_1[idx + 7] = row_0[idx + 7]; 
          }else{
            if(idx + 7 <= capacity) row_1[idx + 7] = row_0[idx + 7] || row_0[idx + 7 - v];
          } 

          if(idx + 8 < v){
            row_1[idx + 8] = row_0[idx + 8]; 
          }else{
            if(idx + 8 <= capacity) row_1[idx + 8] = row_0[idx + 8] || row_0[idx + 8 - v];
          }

          if(idx + 9 < v){
            row_1[idx + 9] = row_0[idx + 9]; 
          }else{
            if(idx + 9 <= capacity) row_1[idx + 9] = row_0[idx + 9] || row_0[idx + 9 - v];
          }

          if(idx + 10 < v){
            row_1[idx + 10] = row_0[idx + 10]; 
          }else{
            if(idx + 10 <= capacity) row_1[idx + 10] = row_0[idx + 10] || row_0[idx + 10 - v];
          }

          if(idx + 11 < v){
            row_1[idx + 11] = row_0[idx + 11]; 
          }else{
            if(idx + 11 <= capacity) row_1[idx + 11] = row_0[idx + 11] || row_0[idx + 11 - v];
          }

          if(idx + 12 < v){
            row_1[idx + 12] = row_0[idx + 12]; 
          }else{
            if(idx + 12 <= capacity) row_1[idx + 12] = row_0[idx + 12] || row_0[idx + 12 - v];
          }

          if(idx + 13 < v){
            row_1[idx + 13] = row_0[idx + 13]; 
          }else{
            if(idx + 13 <= capacity) row_1[idx + 13] = row_0[idx + 13] || row_0[idx + 13 - v];
          }

          if(idx + 14 < v){
            row_1[idx + 14] = row_0[idx + 14]; 
          }else{
            if(idx + 14 <= capacity) row_1[idx + 14] = row_0[idx + 14] || row_0[idx + 14 - v];
          }

          if(idx + 15 < v){
            row_1[idx + 15] = row_0[idx + 15]; 
          }else{
            if(idx + 15 <= capacity) row_1[idx + 15] = row_0[idx + 15] || row_0[idx + 15 - v];
          }
        
      }else{
        //situation 3, because the situation is either 1, 2 or 3.
        if(idx + 15 <= capacity){
          row_1[idx] = row_0[idx] || row_0[idx - v];
          row_1[idx + 1] = row_0[idx + 1] || row_0[idx + 1 - v];
          row_1[idx + 2] = row_0[idx + 2] || row_0[idx + 2 - v];
          row_1[idx + 3] = row_0[idx + 3] || row_0[idx + 3 - v];
          row_1[idx + 4] = row_0[idx + 4] || row_0[idx + 4 - v];
          row_1[idx + 5] = row_0[idx + 5] || row_0[idx + 5 - v];
          row_1[idx + 6] = row_0[idx + 6] || row_0[idx + 6 - v];
          row_1[idx + 7] = row_0[idx + 7] || row_0[idx + 7 - v];
          row_1[idx + 8] = row_0[idx + 8] || row_0[idx + 8 - v];
          row_1[idx + 9] = row_0[idx + 9] || row_0[idx + 9 - v];
          row_1[idx + 10] = row_0[idx + 10] || row_0[idx + 10 - v];
          row_1[idx + 11] = row_0[idx + 11] || row_0[idx + 11 - v];
          row_1[idx + 12] = row_0[idx + 12] || row_0[idx + 12 - v];
          row_1[idx + 13] = row_0[idx + 13] || row_0[idx + 13 - v];
          row_1[idx + 14] = row_0[idx + 14] || row_0[idx + 14 - v];
          row_1[idx + 15] = row_0[idx + 15] || row_0[idx + 15 - v];
        }else{
          if(idx <= capacity)     row_1[idx] = row_0[idx] || row_0[idx - v];
          if(idx + 1 <= capacity) row_1[idx + 1] = row_0[idx + 1] || row_0[idx + 1 - v];
          if(idx + 2 <= capacity) row_1[idx + 2] = row_0[idx + 2] || row_0[idx + 2 - v];
          if(idx + 3 <= capacity) row_1[idx + 3] = row_0[idx + 3] || row_0[idx + 3 - v];
          if(idx + 4 <= capacity) row_1[idx + 4] = row_0[idx + 4] || row_0[idx + 4 - v];
          if(idx + 5 <= capacity) row_1[idx + 5] = row_0[idx + 5] || row_0[idx + 5 - v];
          if(idx + 6 <= capacity) row_1[idx + 6] = row_0[idx + 6] || row_0[idx + 6 - v];
          if(idx + 7 <= capacity) row_1[idx + 7] = row_0[idx + 7] || row_0[idx + 7 - v];
          if(idx + 8 <= capacity) row_1[idx + 8] = row_0[idx + 8] || row_0[idx + 8 - v];
          if(idx + 9 <= capacity) row_1[idx + 9] = row_0[idx + 9] || row_0[idx + 9 - v];
          if(idx + 10 <= capacity) row_1[idx + 10] = row_0[idx + 10] || row_0[idx + 10 - v];
          if(idx + 11 <= capacity) row_1[idx + 11] = row_0[idx + 11] || row_0[idx + 11 - v];
          if(idx + 12 <= capacity) row_1[idx + 12] = row_0[idx + 12] || row_0[idx + 12 - v];
          if(idx + 13 <= capacity) row_1[idx + 13] = row_0[idx + 13] || row_0[idx + 13 - v];
          if(idx + 14 <= capacity) row_1[idx + 14] = row_0[idx + 14] || row_0[idx + 14 - v];
        }
      }
    }

  }else{

    //compute row_0 from row_1

    if(situation == 1){
      row_0[idx] = row_1[idx];
      row_0[idx + 1] = row_1[idx + 1];
      row_0[idx + 2] = row_1[idx + 2];
      row_0[idx + 3] = row_1[idx + 3];
      row_0[idx + 4] = row_1[idx + 4];
      row_0[idx + 5] = row_1[idx + 5];
      row_0[idx + 6] = row_1[idx + 6];
      row_0[idx + 7] = row_1[idx + 7];
      row_0[idx + 8] = row_1[idx + 8];
      row_0[idx + 9] = row_1[idx + 9];
      row_0[idx + 10] = row_1[idx + 10];
      row_0[idx + 11] = row_1[idx + 11];
      row_0[idx + 12] = row_1[idx + 12];
      row_0[idx + 13] = row_1[idx + 13];
      row_0[idx + 14] = row_1[idx + 14];
      row_0[idx + 15] = row_1[idx + 15];
    }else{
      if(situation == 2){
          
          if(idx < v){
            row_0[idx] = row_1[idx]; 
          }else{
            if(idx <= capacity) row_0[idx] = row_1[idx] || row_1[idx - v];
          } 

          if(idx + 1 < v){
            row_0[idx + 1] = row_1[idx + 1]; 
          }else{
            if(idx + 1 <= capacity) row_0[idx + 1] = row_1[idx + 1] || row_1[idx + 1 - v];
          } 

          if(idx + 2 < v){
            row_0[idx + 2] = row_1[idx + 2]; 
          }else{
            if(idx + 2 <= capacity) row_0[idx + 2] = row_1[idx + 2] || row_1[idx + 2 - v];
          } 

          if(idx + 3 < v){
            row_0[idx + 3] = row_1[idx + 3]; 
          }else{
            if(idx + 3 <= capacity) row_0[idx + 3] = row_1[idx + 3] || row_1[idx + 3 - v];
          } 

          if(idx + 4 < v){
            row_0[idx + 4] = row_1[idx + 4]; 
          }else{
            if(idx + 4 <= capacity) row_0[idx + 4] = row_1[idx + 4] || row_1[idx + 4 - v];
          } 

          if(idx + 5 < v){
            row_0[idx + 5] = row_1[idx + 5]; 
          }else{
            if(idx + 5 <= capacity) row_0[idx + 5] = row_1[idx + 5] || row_1[idx + 5 - v];
          } 

          if(idx + 6 < v){
            row_0[idx + 6] = row_1[idx + 6]; 
          }else{
            if(idx + 6 <= capacity) row_0[idx + 6] = row_1[idx + 6] || row_1[idx + 6 - v];
          } 

          if(idx + 7 < v){
            row_0[idx + 7] = row_1[idx + 7]; 
          }else{
            if(idx + 7 <= capacity) row_0[idx + 7] = row_1[idx + 7] || row_1[idx + 7 - v];
          } 

          if(idx + 8 < v){
            row_0[idx + 8] = row_1[idx + 8]; 
          }else{
            if(idx + 8 <= capacity) row_0[idx + 8] = row_1[idx + 8] || row_1[idx + 8 - v];
          }

          if(idx + 9 < v){
            row_0[idx + 9] = row_1[idx + 9]; 
          }else{
            if(idx + 9 <= capacity) row_0[idx + 9] = row_1[idx + 9] || row_1[idx + 9 - v];
          }

          if(idx + 10 < v){
            row_0[idx + 10] = row_1[idx + 10]; 
          }else{
            if(idx + 10 <= capacity) row_0[idx + 10] = row_1[idx + 10] || row_1[idx + 10 - v];
          }

          if(idx + 11 < v){
            row_0[idx + 11] = row_1[idx + 11]; 
          }else{
            if(idx + 11 <= capacity) row_0[idx + 11] = row_1[idx + 11] || row_1[idx + 11 - v];
          }

          if(idx + 12 < v){
            row_0[idx + 12] = row_1[idx + 12]; 
          }else{
            if(idx + 12 <= capacity) row_0[idx + 12] = row_1[idx + 12] || row_1[idx + 12 - v];
          }

          if(idx + 13 < v){
            row_0[idx + 13] = row_1[idx + 13]; 
          }else{
            if(idx + 13 <= capacity) row_0[idx + 13] = row_1[idx + 13] || row_1[idx + 13 - v];
          }

          if(idx + 14 < v){
            row_0[idx + 14] = row_1[idx + 14]; 
          }else{
            if(idx + 14 <= capacity) row_0[idx + 14] = row_1[idx + 14] || row_1[idx + 14 - v];
          }

          if(idx + 15 < v){
            row_0[idx + 15] = row_1[idx + 15]; 
          }else{
            if(idx + 15 <= capacity) row_0[idx + 15] = row_1[idx + 15] || row_1[idx + 15 - v];
          }
        
      }else{
        //situation 3, because the situation is either 1, 2 or 3.
        if(idx + 15 <= capacity){
          row_0[idx] = row_1[idx] || row_1[idx - v];
          row_0[idx + 1] = row_1[idx + 1] || row_1[idx + 1 - v];
          row_0[idx + 2] = row_1[idx + 2] || row_1[idx + 2 - v];
          row_0[idx + 3] = row_1[idx + 3] || row_1[idx + 3 - v];
          row_0[idx + 4] = row_1[idx + 4] || row_1[idx + 4 - v];
          row_0[idx + 5] = row_1[idx + 5] || row_1[idx + 5 - v];
          row_0[idx + 6] = row_1[idx + 6] || row_1[idx + 6 - v];
          row_0[idx + 7] = row_1[idx + 7] || row_1[idx + 7 - v];
          row_0[idx + 8] = row_1[idx + 8] || row_1[idx + 8 - v];
          row_0[idx + 9] = row_1[idx + 9] || row_1[idx + 9 - v];
          row_0[idx + 10] = row_1[idx + 10] || row_1[idx + 10 - v];
          row_0[idx + 11] = row_1[idx + 11] || row_1[idx + 11 - v];
          row_0[idx + 12] = row_1[idx + 12] || row_1[idx + 12 - v];
          row_0[idx + 13] = row_1[idx + 13] || row_1[idx + 13 - v];
          row_0[idx + 14] = row_1[idx + 14] || row_1[idx + 14 - v];
          row_0[idx + 15] = row_1[idx + 15] || row_1[idx + 15 - v];
        }else{
          if(idx <= capacity)     row_0[idx] = row_1[idx] || row_1[idx - v];
          if(idx + 1 <= capacity) row_0[idx + 1] = row_1[idx + 1] || row_1[idx + 1 - v];
          if(idx + 2 <= capacity) row_0[idx + 2] = row_1[idx + 2] || row_1[idx + 2 - v];
          if(idx + 3 <= capacity) row_0[idx + 3] = row_1[idx + 3] || row_1[idx + 3 - v];
          if(idx + 4 <= capacity) row_0[idx + 4] = row_1[idx + 4] || row_1[idx + 4 - v];
          if(idx + 5 <= capacity) row_0[idx + 5] = row_1[idx + 5] || row_1[idx + 5 - v];
          if(idx + 6 <= capacity) row_0[idx + 6] = row_1[idx + 6] || row_1[idx + 6 - v];
          if(idx + 7 <= capacity) row_0[idx + 7] = row_1[idx + 7] || row_1[idx + 7 - v];
          if(idx + 8 <= capacity) row_0[idx + 8] = row_1[idx + 8] || row_1[idx + 8 - v];
          if(idx + 9 <= capacity) row_0[idx + 9] = row_1[idx + 9] || row_1[idx + 9 - v];
          if(idx + 10 <= capacity) row_0[idx + 10] = row_1[idx + 10] || row_1[idx + 10 - v];
          if(idx + 11 <= capacity) row_0[idx + 11] = row_1[idx + 11] || row_1[idx + 11 - v];
          if(idx + 12 <= capacity) row_0[idx + 12] = row_1[idx + 12] || row_1[idx + 12 - v];
          if(idx + 13 <= capacity) row_0[idx + 13] = row_1[idx + 13] || row_1[idx + 13 - v];
          if(idx + 14 <= capacity) row_0[idx + 14] = row_1[idx + 14] || row_1[idx + 14 - v];
        }
      }
    }

  }
}

//Function that uses the two kernels above to solve the SubsetSumDecision problem

unsigned char DP_v2_uroll16_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block){
  unsigned char result = FALSE;
  
  //first step: create the initial row
  kernel_v2_a_uroll16<<<grid, block>>>(old_row_d, capacity);
  CHECK(cudaDeviceSynchronize());

  //second step: compute the new row from the old one
  int old_row_idx = 0;
  for(int r = 0; r < n_items; r++){
    kernel_v2_b_uroll16<<<grid, block>>>(volumes[r], old_row_d, new_row_d, capacity, old_row_idx);
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
