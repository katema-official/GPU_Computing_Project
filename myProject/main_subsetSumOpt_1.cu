#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "common.h"

#define DEBUG_1 0
#define TRUE 1
#define FALSE 0
#define PRINT_SUBSET_SOL 0

#define BLOCK_DIM_X 128

#define ON_MY_PC 1
#define EXECUTE_CPU 0
#define EXECUTE_GPU_INEFFICIENT 0


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------CPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//recursive approach with exhaustive search (for the subsetSumOptimization)
int subsetSumOptimization_recursive(int accum, int* volumes, int capacity, int n_volumes){
  if(n_volumes == 0) return accum;
  
  if(accum + volumes[0] <= capacity){
    int a = subsetSumOptimization_recursive(accum + volumes[0], &(volumes[1]), capacity, n_volumes - 1);
    int b = subsetSumOptimization_recursive(accum, &(volumes[1]), capacity, n_volumes - 1);
    if(a >= b){
      return a;
    }else{
      return b;
    }
  }else{
    return subsetSumOptimization_recursive(accum, &(volumes[1]), capacity, n_volumes - 1);
  }
}

//Proof of correctness: this algorithm takes the volumes, the capacity, the number of volumes and the solution
//found by some other algorithm, and returns the array of elements that sum up to that solution
int* subsetSumOptimization_recursive_solFound(int solution, int* volumes, int volume_occupied, int n_volumes, int idx_current_elem){
  if(volume_occupied > solution) return NULL;

  if(volume_occupied == solution){
    int* sol = (int*) malloc(n_volumes * sizeof(int));
    for(int i = 0; i < n_volumes; i++){
      sol[i] = 0;
    }
    return sol;
  }

  if(n_volumes == idx_current_elem) return NULL;

  int* a_sol = subsetSumOptimization_recursive_solFound(solution, volumes, volume_occupied + volumes[idx_current_elem], n_volumes, idx_current_elem + 1);
  int* b_sol = subsetSumOptimization_recursive_solFound(solution, volumes, volume_occupied, n_volumes, idx_current_elem + 1);

  if((a_sol == NULL) && (b_sol == NULL)){
    return NULL;
  }

  if(a_sol != NULL){
    a_sol[idx_current_elem] = 1;
    return a_sol;
  }

  if(b_sol != NULL){
    b_sol[idx_current_elem] = 0;
    return b_sol;
  }

  return NULL;




}





//recursive approach with exhaustive search (for the subsetSumDecision, it will be useful in Chad Parry's algorithm)
int subsetSumDecision_recursive(int* volumes, int capacity, int n_volumes){
  if(capacity == 0) return TRUE;
  if(n_volumes == 0) return FALSE;

  return 
    subsetSumDecision_recursive(&(volumes[1]), capacity - volumes[0], n_volumes-1) ||
    subsetSumDecision_recursive(&(volumes[1]), capacity, n_volumes-1);

}

//just if someone wants to be sure, here is a version that also gives the elements of the solution
//NOT NULL in solution = this is the solution that solves the problem with TRUE
//NULL in solution: FALSE
int* subsetSumDecision_recursive_solFound(int* volumes, int capacity, int n_volumes, int idx_current_elem){
  if(capacity == 0){
    int* sol = (int*) malloc(n_volumes * sizeof(int));
    for(int i = 0; i < n_volumes; i++){
      sol[i] = 0;
    }
    return sol;
  }

  if(n_volumes == idx_current_elem) return NULL;

  int* a_sol = subsetSumDecision_recursive_solFound(volumes, capacity - volumes[idx_current_elem], n_volumes, idx_current_elem + 1);
  int* b_sol = subsetSumDecision_recursive_solFound(volumes, capacity, n_volumes, idx_current_elem + 1);

  if((a_sol == NULL) && (b_sol == NULL)){
    return NULL;
  }

  if(a_sol != NULL){
    a_sol[idx_current_elem] = 1;
    return a_sol;
  }

  if(b_sol != NULL){
    b_sol[idx_current_elem] = 0;
    return b_sol;
  }

  return NULL;
}





//Dynamic programming approach

//v1: build a dynamic programming matrix of n_items+1 rows and capacity+1 columns.
//Inefficient from memory point of view
//Inefficient because some operations are perfermed even when not necessary

unsigned char subsetSumDecision_DP_v1(int* volumes, int capacity, int n_items){
    unsigned char** B = (unsigned char**) malloc((n_items+1)*sizeof(unsigned char*));
    for(int i = 0; i < n_items+1; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
            printf("Allocation failed v1\n");
        }
    }
    
    //initialization: the subproblems without items and capacity 0 admit a solution
    for(int i = 0; i < n_items+1; i++){
        B[i][0] = TRUE;
    }
    //initialization: the subproblems without items but a bit of capacity don't admit a solution
    for(int i = 1; i < capacity+1; i++){
        B[0][i] = FALSE;
    }

    unsigned char res = 0;

    //now, the value of each cell of each row can be fully determined by the the previous row
    for(int row = 1; row < n_items + 1; row++){
        int volume_row = volumes[row-1];
        for(int col = 1; col < capacity + 1; col++){
            if(col >= volume_row){
              B[row][col] = B[row - 1][col] || B[row - 1][col - volume_row];
            }else{
              B[row][col] = B[row - 1][col];  //copy the previous entry
            }
        }

        if(DEBUG_1) printf("temporary result: %d\n", B[row][capacity]);

        if(B[row][capacity] == 1){
            res = B[row][capacity];
            break;
        }
    }

    for(int i = 0; i < n_items+1; i++){
        free(B[i]);
    }
    free(B);

    return res;
}

//v2: same as before, but using only 2 rows to use less memory. There is however the added complexity of copying the new row in the old one.
//Inefficient because of multiple memory copies
//Still inefficient because some operations are performed even when not necessary

unsigned char subsetSumDecision_DP_v2(int* volumes, int capacity, int n_items){
    unsigned char** B = (unsigned char**) malloc(2*sizeof(unsigned char*));
    for(int i = 0; i < 2; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
          printf("Allocation failed v2\n");
        }
    }
    
    for(int i = 0; i < 2; i++){
        B[i][0] = TRUE;
    }
    for(int i = 1; i < capacity+1; i++){
        B[0][i] = FALSE;
    }

    unsigned char res = 0;

    //now, the value of each cell of each row can be fully determined by the the previous row
    for(int iteration = 0; iteration < n_items; iteration++){
        int volume_row = volumes[iteration];
        for(int col = 1; col < capacity + 1; col++){
            if(col >= volume_row){  //this item could be part of the solution
                B[1][col] = B[0][col] || B[0][col - volume_row];
            }else{
                B[1][col] = B[0][col];  //the volume of this item is more than the current capacity
            }
        }

        //now copy the new row in the old one
        for(int col = 1; col < capacity + 1; col++){
            B[0][col] = B[1][col];
        }

        if(DEBUG_1) printf("temporary result: %d\n", B[0][capacity]);

        if(B[0][capacity] == 1){
            res = B[1][capacity];
            break;
        }

    }

    for(int i = 0; i < 2; i++){
        free(B[i]);
    }
    free(B);

    return res;
}

//v3: doing everything in one row. There is no overhead because of copy operations.
//We also avoid performing useless operations.

unsigned char subsetSumDecision_DP_v3(int* volumes, int capacity, int n_items){
    if(capacity == 0 || n_items == 0){
        return 0;
    }
    unsigned char* B = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
    for(int i = capacity; i > 0; i--){
        B[i] = FALSE;
    }
    B[0] = TRUE;

    unsigned char res = 0;

    //now, the value of each cell of each row can be fully determined by the the previous row,
    //that is actually the same row
    for(int iteration = 0; iteration < n_items; iteration++){
        int volume_row = volumes[iteration];
        for(int col = capacity; col >=volume_row; col--){
            B[col] = B[col] || B[col - volume_row];
            //printf("B[%d] = %d ", col, B[col]);
        }

        if(DEBUG_1) printf("temporary result: %d\n", B[capacity]);

        if(B[capacity] == 1){
            res = B[capacity];
            break;
        }
    }

    free(B);
    return res;
}




int cmpfunc_increasing(const void * a, const void * b) {
   return (*(int*)a - *(int*)b);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------GPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

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





//v2 uroll8: as v2, but we give more work to each single thread with the unrolling technique

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









//to run as:
// ./main [random_or_not] [n_vols] [capacity] [random_seed] [blockDim.x]

int main(int argc, char **argv){

  int n_vols = 32;
  int* vols;
  int capacity = 10000;//12345678;

  //the first arguments tells if the sequence of volumes must be randomly generated (1)
  //or not (0)
  int generate_randomly_flag = 0;
  if(argc > 1){
    generate_randomly_flag = atoi(argv[1]);
  }

  //the second argument is the number of volumes. If 0, the default one is used.
  if(argc > 2){
    int _n_vols = atoi(argv[2]);
    if(_n_vols > 0){
      n_vols = _n_vols;
    }
  }
  vols = (int*) malloc(n_vols * sizeof(int));

  //the third argument is the total capacity. If 0, the default one is used.
  if(argc > 3){
    int _capacity = atoi(argv[3]);
    if(_capacity > 0){
      capacity = _capacity;
    }
  }

  //the fourth argument is the seed to be used in case of randomly generated volumes.
  //if 0, then the seed is randomized. Otherwise, the argument becomes the seed.

  if(generate_randomly_flag){
    int seed = 0;
    srand(time(0));
    if(argc > 4){
      seed = atoi(argv[4]);
      if(seed != 0){
        srand(seed);
      }
    }
    
    //"standard" values:
    //-lower = 50 ==> lower = capacity/200 
    //-upper = 500 ==> upper = capacity/20
    //-capacity = 10000;

    int lower = capacity/1000;
    int upper = capacity/10;
    for(int i = 0; i < n_vols; i++){
      vols[i] = (rand() % (upper - lower + 1) + lower);
      //printf("vols[%d] = %d\n", i, vols[i]);
    }

    //printf just to make sure the seed is correct during multiple runs
    printf("vols[%d] = %d\n", n_vols-1, vols[n_vols-1]);
  }else{
    for(int i = 0; i < n_vols; i++){
      vols[i] = 100*i;
    }
  }

  //check the volumes
  if(DEBUG_1){
    for(int i = 0; i < n_vols; i++){
      printf("vols[%d] = %d\n", i, vols[i]);
    }
  }
  

  
  //----------------------------------------------------------------------------
  //-------------------------------CPU ALGORITHMS-------------------------------
  //----------------------------------------------------------------------------
  
  //Credits for the original Subset Sum Optimization algorithm to Chad Parry:
  //https://courses.cs.washington.edu/courses/csep521/05sp/lectures/sss.pdf
  
  double start, end;

  //worst strategy but surely exact: exhaustive search
  if(EXECUTE_CPU){
    start = seconds();
    int result_exhaustive = subsetSumOptimization_recursive(0, vols, capacity, n_vols);
    end = seconds() - start;
    if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
    printf("recursive exhaustive CPU optimization, res: %d, elapsed: %f\n", result_exhaustive, end * 1000);

    //if one has enough time, with this call it can see the actual solution (the optimal subset)
    if(PRINT_SUBSET_SOL){
      int* subset;
      subset = subsetSumOptimization_recursive_solFound(result_exhaustive, vols, 0, n_vols, 0);
      for(int i = 0; i < n_vols; i++){
        if(subset[i] == 1){
          printf("%d + ", vols[i]);
        }
      }
      printf(" = %d\n", result_exhaustive);
    }
  }


  //first of all, following the algorithm, we have to agument the set of volumes with all the
  //powers of 2 smaller than capacity

  int* vols_agumented = (int*) malloc((n_vols + log2(capacity) + 1) * sizeof(int));
  for(int i = 0; i < n_vols; i++){
    vols_agumented[i] = vols[i];
  }

  for(int i = log2(capacity); i >= 0; i--){
    int add = pow(2, i);
    vols_agumented[n_vols + i] = add;
  }
  

  int res = capacity;
  int n_vols_agumented = n_vols + (int) log2(capacity) + 1;

  //don't remove the "&& 0", this algorithm takes waaaaay too long to complete
  if(EXECUTE_CPU && 0){
    start = seconds();
    for(int i = 0; i <= (int) log2(capacity); i++){
      n_vols_agumented--;   //simulates A <- A \ 2^i
      if(subsetSumDecision_recursive(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds() - start;
    if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
    printf("recursive exhaustive CPU Chad Parry, res: %d, elapsed: %f\n", res, end * 1000);
  }


  if(EXECUTE_CPU){
    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;
    for(int i = 0; i <= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(subsetSumDecision_DP_v1(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds() - start;
    if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
    printf("DP v1 CPU, res: %d, elapsed: %f\n", res, end * 1000);
  }


  if(EXECUTE_CPU){
    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;
    for(int i = 0; i<= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(subsetSumDecision_DP_v2(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds() - start;
    if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
    printf("DP v2 CPU, res: %d, elapsed: %f\n", res, end * 1000);
  }


  if(EXECUTE_CPU){
    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;
    for(int i = 0; i<= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(subsetSumDecision_DP_v3(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds() - start;
    if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
    printf("DP v3 CPU, res: %d, elapsed: %f\n", res, end * 1000);
  }


  //----------------------------------------------------------------------------
  //-------------------------------GPU ALGORITHMS-------------------------------
  //----------------------------------------------------------------------------

  //if one wants to change the division between L1 cache and shared memory,
  //giving more shared memory for the smem approach or more L1 cache for the
  //kernels that don't use it
  CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  dim3 block(BLOCK_DIM_X);
  dim3 grid(((capacity + 1) + block.x - 1)/block.x);
  printf("block.x = %d, grid.x = %d\n", block.x, grid.x);

  //first, we need to declare host and device memory, and initialize it
  
  unsigned char* row_h = (unsigned char*) malloc((capacity + 1) * sizeof(unsigned char));
  if(row_h == NULL){
    printf("allocation failed!\n");
  }

  unsigned char *old_row_d, *new_row_d;
  CHECK(cudaMalloc((unsigned char**)&old_row_d, (capacity + 1) * sizeof(unsigned char)));
  CHECK(cudaMalloc((unsigned char**)&new_row_d, (capacity + 1) * sizeof(unsigned char)));
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t eStart, eEnd;
  float msEvent;

  //-----------------------------------GPU v1-----------------------------------

  if(EXECUTE_GPU_INEFFICIENT){
    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;

    CHECK(cudaEventCreate(&eStart));
    CHECK(cudaEventCreate(&eEnd));
    CHECK(cudaEventRecord(eStart, 0));

    for(int i = 0; i<= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(DP_v1_GPU(vols_agumented, res, n_vols_agumented, row_h, old_row_d, new_row_d, grid, block) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }

    CHECK(cudaEventRecord(eEnd, 0));
    CHECK(cudaEventSynchronize(eEnd));

    CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

    end = seconds() - start;
    if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
    if(ON_MY_PC) msEvent = msEvent / 1000;
    printf("DP v1 GPU, res: %d, elapsed: %f, event time: %f\n", res, end * 1000, msEvent);

    CHECK(cudaEventDestroy(eStart));
    CHECK(cudaEventDestroy(eEnd));
  }

  //-----------------------------------GPU v2-----------------------------------

  start = seconds();
  res = capacity;
  n_vols_agumented = n_vols + (int) log2(capacity) + 1;

  CHECK(cudaEventCreate(&eStart));
  CHECK(cudaEventCreate(&eEnd));
  CHECK(cudaEventRecord(eStart, 0));

  for(int i = 0; i<= (int) log2(capacity); i++){
    n_vols_agumented--;
    if(DP_v2_GPU(vols_agumented, res, n_vols_agumented, row_h, old_row_d, new_row_d, grid, block) == FALSE){
      res = res - vols_agumented[n_vols_agumented];
    }
  }

  CHECK(cudaEventRecord(eEnd, 0));
  CHECK(cudaEventSynchronize(eEnd));

  CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

  end = seconds() - start;
  if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
  if(ON_MY_PC) msEvent = msEvent / 1000;
  printf("DP v2 GPU, res: %d, elapsed: %f, event time: %f\n", res, end * 1000, msEvent);

  CHECK(cudaEventDestroy(eStart));
  CHECK(cudaEventDestroy(eEnd));


  //---------------------------------GPU v2 smem---------------------------------
  CHECK(cudaFuncSetCacheConfig(kernel_v2_a_smem, cudaFuncCachePreferShared));
  CHECK(cudaFuncSetCacheConfig(kernel_v2_b_smem, cudaFuncCachePreferShared));

  start = seconds();
  res = capacity;
  n_vols_agumented = n_vols + (int) log2(capacity) + 1;

  CHECK(cudaEventCreate(&eStart));
  CHECK(cudaEventCreate(&eEnd));
  CHECK(cudaEventRecord(eStart, 0));

  for(int i = 0; i<= (int) log2(capacity); i++){
    n_vols_agumented--;
    if(DP_v2_smem_GPU(vols_agumented, res, n_vols_agumented, row_h, old_row_d, new_row_d, grid, block) == FALSE){
      res = res - vols_agumented[n_vols_agumented];
    }
  }

  CHECK(cudaEventRecord(eEnd, 0));
  CHECK(cudaEventSynchronize(eEnd));

  CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

  end = seconds() - start;
  if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
  if(ON_MY_PC) msEvent = msEvent / 1000;
  printf("DP v2 smem GPU, res: %d, elapsed: %f, event time: %f\n", res, end * 1000, msEvent);

  CHECK(cudaEventDestroy(eStart));
  CHECK(cudaEventDestroy(eEnd));

  //--------------------------------GPU v2 uroll8--------------------------------
  CHECK(cudaFuncSetCacheConfig(kernel_v2_a_smem, cudaFuncCachePreferShared));
  CHECK(cudaFuncSetCacheConfig(kernel_v2_b_smem, cudaFuncCachePreferShared));
  
  dim3 grid8((grid.x + 8 - 1) / 8);
  start = seconds();
  res = capacity;
  n_vols_agumented = n_vols + (int) log2(capacity) + 1;

  CHECK(cudaEventCreate(&eStart));
  CHECK(cudaEventCreate(&eEnd));
  CHECK(cudaEventRecord(eStart, 0));

  for(int i = 0; i<= (int) log2(capacity); i++){
    n_vols_agumented--;
    if(DP_v2_uroll8_GPU(vols_agumented, res, n_vols_agumented, row_h, old_row_d, new_row_d, grid8, block) == FALSE){
      res = res - vols_agumented[n_vols_agumented];
    }
  }

  CHECK(cudaEventRecord(eEnd, 0));
  CHECK(cudaEventSynchronize(eEnd));

  CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

  end = seconds() - start;
  if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
  if(ON_MY_PC) msEvent = msEvent / 1000;
  printf("DP v2 uroll8 GPU, res: %d, elapsed: %f, event time: %f\n", res, end * 1000, msEvent);

  CHECK(cudaEventDestroy(eStart));
  CHECK(cudaEventDestroy(eEnd));

  //------------------------------GPU v2 uroll8 smem------------------------------
  CHECK(cudaFuncSetCacheConfig(kernel_v2_a_uroll8_smem, cudaFuncCachePreferShared));
  CHECK(cudaFuncSetCacheConfig(kernel_v2_b_uroll8_smem, cudaFuncCachePreferShared));

  start = seconds();
  res = capacity;
  n_vols_agumented = n_vols + (int) log2(capacity) + 1;

  CHECK(cudaEventCreate(&eStart));
  CHECK(cudaEventCreate(&eEnd));
  CHECK(cudaEventRecord(eStart, 0));

  for(int i = 0; i<= (int) log2(capacity); i++){
    n_vols_agumented--;
    if(DP_v2_uroll8_smem_GPU(vols_agumented, res, n_vols_agumented, row_h, old_row_d, new_row_d, grid8, block) == FALSE){
      res = res - vols_agumented[n_vols_agumented];
    }
  }

  CHECK(cudaEventRecord(eEnd, 0));
  CHECK(cudaEventSynchronize(eEnd));

  CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

  end = seconds() - start;
  if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
  if(ON_MY_PC) msEvent = msEvent / 1000;
  printf("DP v2 uroll8 smem GPU, res: %d, elapsed: %f, event time: %f\n", res, end * 1000, msEvent);

  CHECK(cudaEventDestroy(eStart));
  CHECK(cudaEventDestroy(eEnd));

  //--------------------------------GPU v2 uroll16--------------------------------
  
  dim3 grid16((grid.x + 16 - 1) / 16);
  start = seconds();
  res = capacity;
  n_vols_agumented = n_vols + (int) log2(capacity) + 1;

  CHECK(cudaEventCreate(&eStart));
  CHECK(cudaEventCreate(&eEnd));
  CHECK(cudaEventRecord(eStart, 0));

  for(int i = 0; i<= (int) log2(capacity); i++){
    n_vols_agumented--;
    if(DP_v2_uroll16_GPU(vols_agumented, res, n_vols_agumented, row_h, old_row_d, new_row_d, grid16, block) == FALSE){
      res = res - vols_agumented[n_vols_agumented];
    }
  }

  CHECK(cudaEventRecord(eEnd, 0));
  CHECK(cudaEventSynchronize(eEnd));

  CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

  end = seconds() - start;
  if(ON_MY_PC) end = (end / CLOCKS_PER_SEC) / 1000;
  if(ON_MY_PC) msEvent = msEvent / 1000;
  printf("DP v2 uroll16 GPU, res: %d, elapsed: %f, event time: %f\n", res, end * 1000, msEvent);

  CHECK(cudaEventDestroy(eStart));
  CHECK(cudaEventDestroy(eEnd));







  //finally, release the memory

  

  free(row_h);

  CHECK(cudaFree(old_row_d));
  CHECK(cudaFree(new_row_d));
  CHECK(cudaDeviceReset());








  





  free(vols_agumented);
  free(vols);


}


