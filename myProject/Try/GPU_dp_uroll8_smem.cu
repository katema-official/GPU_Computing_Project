#include "common.h"
#include <stdio.h>

//v2 uroll8 smem: as v2, but we give more work to each single thread with the unrolling technique, and we leverage on the shared memory

__global__ void kernel_v2_a_uroll8_smem(unsigned char* row_0, int capacity){
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

__global__ void kernel_v2_b_uroll8_smem(int v, unsigned char* row_0, unsigned char* row_1, int capacity, int old_row_idx){
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

  __shared__ unsigned char smem[BLOCK_DIM_X * 2 * 8];

  unsigned char a[8];
  unsigned char b[8];

  int warp_id = threadIdx.x / 32;
  int tid_in_warp = threadIdx.x % 32;

  
  if(old_row_idx == 0){

    //compute row_1 from row_0

    if(situation == 1){
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_0[idx + 0];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_0[idx + 1];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_0[idx + 2];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_0[idx + 3];
            
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_0[idx + 4];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_0[idx + 5];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_0[idx + 6];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_0[idx + 7];



      a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
      a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
      a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
      a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];

      a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
      a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
      a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
      a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];



      row_1[idx + 0] = a[0];
      row_1[idx + 1] = a[1];
      row_1[idx + 2] = a[2];
      row_1[idx + 3] = a[3];

      row_1[idx + 4] = a[4];
      row_1[idx + 5] = a[5];  
      row_1[idx + 6] = a[6]; 
      row_1[idx + 7] = a[7]; 

    }else{
      if(situation == 2){
          
          if(idx + 0 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_0[idx + 0];
            a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
            row_1[idx + 0] = a[0];
          }else{
            if(idx + 0 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_0[idx + 0];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1] = row_0[idx + 0 - v];
              a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
              b[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1];
              row_1[idx + 0] = a[0] || b[0];
            }
          } 

          if(idx + 1 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_0[idx + 1];
            a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
            row_1[idx + 1] = a[1];
          }else{
            if(idx + 1 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_0[idx + 1];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1] = row_0[idx + 1 - v];
              a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
              b[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1];
              row_1[idx + 1] = a[1] || b[1];
            }
          } 

          if(idx + 2 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_0[idx + 2];
            a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
            row_1[idx + 2] = a[2];
          }else{
            if(idx + 2 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_0[idx + 2];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1] = row_0[idx + 2 - v];
              a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
              b[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1];
              row_1[idx + 2] = a[2] || b[2];
            }
          } 

          if(idx + 3 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_0[idx + 3];
            a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];
            row_1[idx + 3] = a[3];
          }else{
            if(idx + 3 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_0[idx + 3];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1] = row_0[idx + 3 - v];
              a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];
              b[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1];
              row_1[idx + 3] = a[3] || b[3];
            }
          } 

          if(idx + 4 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_0[idx + 4];
            a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
            row_1[idx + 4] = a[4];
          }else{
            if(idx + 4 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_0[idx + 4];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3] = row_0[idx + 4 - v];
              a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
              b[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3];
              row_1[idx + 4] = a[4] || b[4];
            }
          } 

          if(idx + 5 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_0[idx + 5];
            a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
            row_1[idx + 5] = a[5];
          }else{
            if(idx + 5 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_0[idx + 5];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3] = row_0[idx + 5 - v];
              a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
              b[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3];
              row_1[idx + 5] = a[5] || b[5];
            }
          } 

          if(idx + 6 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_0[idx + 6];
            a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
            row_1[idx + 6] = a[6]; 
          }else{
            if(idx + 6 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_0[idx + 6];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3] = row_0[idx + 6 - v];
              a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
              b[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3];
              row_1[idx + 6] = a[6] || b[6];
            }
          } 

          if(idx + 7 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_0[idx + 7];
            a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];
            row_1[idx + 7] = a[7]; 
          }else{
            if(idx + 7 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_0[idx + 7];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3] = row_0[idx + 7 - v];
              a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];
              b[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3];
              row_1[idx + 7] = a[7] || b[7];
            }
          } 
        
      }else{
        //situation 3, because the situation is either 1, 2 or 3.
        if(idx + 7 <= capacity){
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_0[idx + 0];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_0[idx + 1];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_0[idx + 2];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_0[idx + 3];

          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1] = row_0[idx + 0 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1] = row_0[idx + 1 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1] = row_0[idx + 2 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1] = row_0[idx + 3 - v];

          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_0[idx + 4];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_0[idx + 5];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_0[idx + 6];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_0[idx + 7];

          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3] = row_0[idx + 4 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3] = row_0[idx + 5 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3] = row_0[idx + 6 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3] = row_0[idx + 7 - v];

          a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
          a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
          a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
          a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];

          b[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1];
          b[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1];
          b[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1];
          b[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1];

          a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
          a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
          a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
          a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];

          b[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3];
          b[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3];
          b[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3];
          b[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3];

          row_1[idx + 0] = a[0] || b[0];
          row_1[idx + 1] = a[1] || b[1];
          row_1[idx + 2] = a[2] || b[2];
          row_1[idx + 3] = a[3] || b[3];
          row_1[idx + 4] = a[4] || b[4];
          row_1[idx + 5] = a[5] || b[5];
          row_1[idx + 6] = a[6] || b[6];
          row_1[idx + 7] = a[7] || b[7];

        }else{
          if(idx + 0 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_0[idx + 0];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1] = row_0[idx + 0 - v];
            a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
            b[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1];
            row_1[idx + 0] = a[0] || b[0];
          }

          if(idx + 1 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_0[idx + 1];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1] = row_0[idx + 1 - v];
            a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
            b[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1];
            row_1[idx + 1] = a[1] || b[1];
          }

          if(idx + 2 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_0[idx + 2];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1] = row_0[idx + 2 - v];
            a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
            b[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1];
            row_1[idx + 2] = a[2] || b[2];
          }

          if(idx + 3 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_0[idx + 3];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1] = row_0[idx + 3 - v];
            a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];
            b[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1];
            row_1[idx + 3] = a[3] || b[3];
          } 
          
          if(idx + 4 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_0[idx + 4];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3] = row_0[idx + 4 - v];
            a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
            b[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3];
            row_1[idx + 4] = a[4] || b[4];
          }
          
          if(idx + 5 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_0[idx + 5];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3] = row_0[idx + 5 - v];
            a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
            b[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3];
            row_1[idx + 5] = a[5] || b[5];
          } 
          
          if(idx + 6 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_0[idx + 6];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3] = row_0[idx + 6 - v];
            a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
            b[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3];
            row_1[idx + 6] = a[6] || b[6];
          }
        }
      }
    }

  }else{

    //compute row_0 from row_1

    if(situation == 1){

      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_1[idx + 0];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_1[idx + 1];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_1[idx + 2];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_1[idx + 3];
            
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_1[idx + 4];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_1[idx + 5];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_1[idx + 6];
      smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_1[idx + 7];



      a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
      a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
      a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
      a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];

      a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
      a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
      a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
      a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];



      row_0[idx + 0] = a[0];
      row_0[idx + 1] = a[1];
      row_0[idx + 2] = a[2];
      row_0[idx + 3] = a[3];

      row_0[idx + 4] = a[4];
      row_0[idx + 5] = a[5];  
      row_0[idx + 6] = a[6]; 
      row_0[idx + 7] = a[7]; 

    }else{
      if(situation == 2){
          
          if(idx + 0 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_1[idx + 0];
            a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
            row_0[idx + 0] = a[0];
          }else{
            if(idx + 0 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_1[idx + 0];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1] = row_1[idx + 0 - v];
              a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
              b[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1];
              row_0[idx + 0] = a[0] || b[0];
            }
          } 

          if(idx + 1 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_1[idx + 1];
            a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
            row_0[idx + 1] = a[1];
          }else{
            if(idx + 1 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_1[idx + 1];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1] = row_1[idx + 1 - v];
              a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
              b[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1];
              row_0[idx + 1] = a[1] || b[1];
            }
          } 

          if(idx + 2 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_1[idx + 2];
            a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
            row_0[idx + 2] = a[2];
          }else{
            if(idx + 2 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_1[idx + 2];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1] = row_1[idx + 2 - v];
              a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
              b[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1];
              row_0[idx + 2] = a[2] || b[2];
            }
          } 

          if(idx + 3 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_1[idx + 3];
            a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];
            row_0[idx + 3] = a[3];
          }else{
            if(idx + 3 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_1[idx + 3];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1] = row_1[idx + 3 - v];
              a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];
              b[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1];
              row_0[idx + 3] = a[3] || b[3];
            }
          } 

          if(idx + 4 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_1[idx + 4];
            a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
            row_0[idx + 4] = a[4];
          }else{
            if(idx + 4 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_1[idx + 4];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3] = row_1[idx + 4 - v];
              a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
              b[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3];
              row_0[idx + 4] = a[4] || b[4];
            }
          } 

          if(idx + 5 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_1[idx + 5];
            a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
            row_0[idx + 5] = a[5];
          }else{
            if(idx + 5 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_1[idx + 5];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3] = row_1[idx + 5 - v];
              a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
              b[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3];
              row_0[idx + 5] = a[5] || b[5];
            }
          } 

          if(idx + 6 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_1[idx + 6];
            a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
            row_0[idx + 6] = a[6]; 
          }else{
            if(idx + 6 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_1[idx + 6];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3] = row_1[idx + 6 - v];
              a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
              b[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3];
              row_0[idx + 6] = a[6] || b[6];
            }
          } 

          if(idx + 7 < v){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_1[idx + 7];
            a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];
            row_0[idx + 7] = a[7]; 
          }else{
            if(idx + 7 <= capacity){
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_1[idx + 7];
              smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3] = row_1[idx + 7 - v];
              a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];
              b[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3];
              row_0[idx + 7] = a[7] || b[7];
            }
          } 
        
      }else{
        //situation 3, because the situation is either 1, 2 or 3.
        if(idx + 7 <= capacity){
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_1[idx + 0];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_1[idx + 1];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_1[idx + 2];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_1[idx + 3];

          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1] = row_1[idx + 0 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1] = row_1[idx + 1 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1] = row_1[idx + 2 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1] = row_1[idx + 3 - v];

          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_1[idx + 4];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_1[idx + 5];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_1[idx + 6];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2] = row_1[idx + 7];

          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3] = row_1[idx + 4 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3] = row_1[idx + 5 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3] = row_1[idx + 6 - v];
          smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3] = row_1[idx + 7 - v];

          a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
          a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
          a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
          a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];

          b[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1];
          b[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1];
          b[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1];
          b[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1];

          a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
          a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
          a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
          a[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 2];

          b[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3];
          b[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3];
          b[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3];
          b[7] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 3];

          row_0[idx + 0] = a[0] || b[0];
          row_0[idx + 1] = a[1] || b[1];
          row_0[idx + 2] = a[2] || b[2];
          row_0[idx + 3] = a[3] || b[3];
          row_0[idx + 4] = a[4] || b[4];
          row_0[idx + 5] = a[5] || b[5];
          row_0[idx + 6] = a[6] || b[6];
          row_0[idx + 7] = a[7] || b[7];

        }else{
          if(idx + 0 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0] = row_1[idx + 0];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1] = row_1[idx + 0 - v];
            a[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 0];
            b[0] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 1];
            row_0[idx + 0] = a[0] || b[0];
          }

          if(idx + 1 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0] = row_1[idx + 1];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1] = row_1[idx + 1 - v];
            a[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 0];
            b[1] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 1];
            row_0[idx + 1] = a[1] || b[1];
          }

          if(idx + 2 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0] = row_1[idx + 2];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1] = row_1[idx + 2 - v];
            a[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 0];
            b[2] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 1];
            row_0[idx + 2] = a[2] || b[2];
          }

          if(idx + 3 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0] = row_1[idx + 3];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1] = row_1[idx + 3 - v];
            a[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 0];
            b[3] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 3 + 128 * 1];
            row_0[idx + 3] = a[3] || b[3];
          } 
          
          if(idx + 4 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2] = row_1[idx + 4];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3] = row_1[idx + 4 - v];
            a[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 2];
            b[4] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 0 + 128 * 3];
            row_0[idx + 4] = a[4] || b[4];
          }
          
          if(idx + 5 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2] = row_1[idx + 5];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3] = row_1[idx + 5 - v];
            a[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 2];
            b[5] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 1 + 128 * 3];
            row_0[idx + 5] = a[5] || b[5];
          } 
          
          if(idx + 6 <= capacity){
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2] = row_1[idx + 6];
            smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3] = row_1[idx + 6 - v];
            a[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 2];
            b[6] = smem[warp_id * 32 * 8 * 2 + tid_in_warp * 4 + 2 + 128 * 3];
            row_0[idx + 6] = a[6] || b[6];
          }
        }
      }
    }

  }
}

//Function that uses the two kernels above to solve the SubsetSumDecision problem

unsigned char DP_v2_uroll8_smem_GPU(int* volumes, int capacity, int n_items, unsigned char* row_h, unsigned char* old_row_d, unsigned char* new_row_d, dim3 grid, dim3 block){
  unsigned char result = FALSE;
  
  //first step: create the initial row
  kernel_v2_a_uroll8_smem<<<grid, block>>>(old_row_d, capacity);
  CHECK(cudaDeviceSynchronize());

  //second step: compute the new row from the old one
  int old_row_idx = 0;
  for(int r = 0; r < n_items; r++){
    kernel_v2_b_uroll8_smem<<<grid, block>>>(volumes[r], old_row_d, new_row_d, capacity, old_row_idx);
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