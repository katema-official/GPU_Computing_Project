#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../common.h"

#define max(a,b) (((a) > (b)) ? (a) : (b))


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------CPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//v1: build a dynamic programming matrix of n_items rows and capacity+1 columns.
//Inefficient from memory point of view
//Inefficient because some operations are perfermed even when not necessary

int solve_knapsack_v1(int* volumes, int n_items, int capacity){
    int** B = (int**) malloc((n_items+1)*sizeof(int*));
    for(int i = 0; i < n_items+1; i++){
        B[i] = (int*) malloc((capacity+1)*sizeof(int));
        if(B[i] == NULL){
            printf("Allocation failed\n");
        }
    }
    
    //initialization: the subproblems without items or capacity have as best solution 0
    for(int i = 0; i < n_items+1; i++){
        B[i][0] = 0;
    }
    for(int i = 0; i < capacity+1; i++){
        B[0][i] = 0;
    }
    printf("aaa1\n");

    //now, the value of each cell of each row can be fully determined by the the previous row
    for(int row = 1; row < n_items + 1; row++){
        int volume_row = volumes[row-1];
        for(int col = 1; col < capacity + 1; col++){
            if(volume_row <= col){  //this item could be part of the solution
                if((volume_row + B[row-1][col - volume_row]) > B[row-1][col]){
                    B[row][col] = volume_row + B[row-1][col - volume_row];
                }else{
                    B[row][col] = B[row-1][col];
                }
            }else{
                B[row][col] = B[row - 1][col];  //the volume of this item is more than the current capacity
            }
        }
    }

    int res = B[n_items][capacity];
    printf("aaa2\n");

    for(int i = 0; i < n_items+1; i++){
        free(B[i]);
    }
    free(B);

    return res;
}





//v2: same as before, but using only 2 rows to use less memory. There is however the added complexity of copying the new row in the old one.
//Inefficient because of multiple memory copies
//Still inefficient because some operations are performed even when not necessary

int solve_knapsack_v2(int* volumes, int n_items, int capacity){
    int** B = (int**) malloc(2*sizeof(int*));
    for(int i = 0; i < 2; i++){
        B[i] = (int*) malloc((capacity+1)*sizeof(int));
        if(B[i] == NULL){
            printf("Allocation failed\n");
        }
    }
    
    //initialization: the subproblems without items or capacity have as best solution 0
    for(int i = 0; i < 2; i++){
        B[i][0] = 0;
    }
    for(int i = 0; i < capacity+1; i++){
        B[0][i] = 0;
    }

    //now, the value of each cell of each row can be fully determined by the the previous row
    for(int iteration = 0; iteration < n_items; iteration++){
        int volume_row = volumes[iteration];
        for(int col = 1; col < capacity + 1; col++){
            if(volume_row <= col){  //this item could be part of the solution
                B[1][col] = max(volume_row + B[0][col - volume_row], B[0][col]);
            }else{
                B[1][col] = B[0][col];  //the volume of this item is more than the current capacity
            }
        }

        //now copy the new row in the old one
        for(int col = 1; col < capacity + 1; col++){
            B[0][col] = B[1][col];
        }

    }

    int res = B[1][capacity];

    for(int i = 0; i < 2; i++){
        free(B[i]);
    }
    free(B);

    return res;
}





//v3: doing everything in one row. There is no overhead because of copy operations
//Inefficient because some of the last operations could still be avoided
//side note: might work more efficiently if the elements are already ordered

int solve_knapsack_v3(int* volumes, int n_items, int capacity){
    if(capacity == 0 || n_items == 0){
        return 0;
    }
    int* B = (int*) malloc((capacity+1)*sizeof(int));
    for(int i = capacity; i >= volumes[0]; i--){
        B[i] = volumes[0];
    }
    for(int i = volumes[0]-1; i >= 0; i--){
        B[i] = 0;
    }

    //now, the value of each cell of each row can be fully determined by the the previous row,
    //that is actually the same row
    for(int iteration = 1; iteration < n_items; iteration++){
        int volume_row = volumes[iteration];
        for(int col = capacity; col >=0; col--){
            if(col >= volume_row){
                B[col] = max(volume_row + B[col - volume_row], B[col]);
            }//else don't do anything, no need to update.
        }
    }

    int res = B[capacity];
    free(B);
    return res;
}





//v4: the last elements require less calculations
int solve_knapsack_v4(int* volumes, int n_items, int capacity){
    if(capacity == 0 || n_items == 0){
        return 0;
    }
    int* B = (int*) malloc((capacity+1)*sizeof(int));
    for(int i = capacity; i >= volumes[0]; i--){
        B[i] = volumes[0];
    }
    for(int i = (volumes[0]-1); i >= 0; i--){
        B[i] = 0;
    }

    //now, the value of each cell of each row can be fully determined by the the previous row,
    //that is actually the same row
    for(int iteration = 1; iteration < n_items; iteration++){
        int capacity_copy = capacity;
        int min_index = 0;
        for(int i = iteration+1; i < n_items; i++){
            capacity_copy -= volumes[i];
        }

        if(capacity_copy <= 0){
            min_index = 0;
        }else{
            min_index = capacity_copy;
        }

        int volume_row = volumes[iteration];
        for(int col = capacity; col >=min_index; col--){
            if(col >= volume_row){
                B[col] = max(volume_row + B[col - volume_row], B[col]);
            }//else don't do anything, no need to update.
        }
    }

    int res = B[capacity];
    free(B);
    return res;
}



int cmpfunc_increasing(const void * a, const void * b) {
   return (*(int*)a - *(int*)b);
}

int cmpfunc_decreasing(const void * a, const void * b) {
   return (*(int*)b - *(int*)a);
}





//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------GPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

__global__ void kernel_v1_a(int v, int* res_row, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < v){
    //the first item can't be placed if the current capacity is less than its volume
    res_row[idx] = 0;
  }else{
    if(idx <= capacity + 1){
      //just place the item
      res_row[idx] = v;
    }
  }
}

__global__ void kernel_v1_b(int v, int* input_row, int* output_row, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //each thread needs to put in the corrisponding cell (given by idx) the max
  //value between:
  //-the current one and the one where 
  //-the value of this capacity minus v, plus v
  
  if(idx >= v){
    //we can start thinking of placing the item only when there is enough
    //current capacity
    output_row[idx] = max(v + input_row[idx - v], input_row[idx]);
  }else{
    if(idx <= capacity + 1){
      //for the first elements, just copy the previous entry
      output_row[idx] = input_row[idx];
    }
  }
}





//to run as:
// ./main [random_or_not] [n_vols] [capacity] [random_seed]

int main(int argc, char **argv){

  

  int n_vols = 32;
  int vols[n_vols];
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
    //-lower = 50
    //-upper = 500
    //-capacity = 10000;

    int lower = capacity/200;
    int upper = capacity/20;
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


  
  //----------------------------------------------------------------------------
  //-------------------------------CPU ALGORITHMS-------------------------------
  //----------------------------------------------------------------------------
  
  double start, end;

  start = seconds();
  int res = solve_knapsack_v1(vols, n_vols, capacity);
  end = seconds() - start;
  printf("v1 CPU, res: %d, elapsed: %f\n", res, end * 1000);


  start = seconds();
  res = solve_knapsack_v2(vols, n_vols, capacity);
  end = seconds() - start;
  printf("v2 CPU, res: %d, elapsed: %f\n", res, end * 1000);


  start = seconds();
  res = solve_knapsack_v3(vols, n_vols, capacity);
  end = seconds() - start;
  printf("v3 CPU, res: %d, elapsed: %f\n", res, end * 1000);

  //let's try with the ordered volumes to see if we have a speedup
  int vols_ordered_increasing[n_vols];
  for(int i = 0; i < n_vols; i++){
    vols_ordered_increasing[i] = vols[i];
  }
  qsort(vols_ordered_increasing, n_vols, sizeof(int), cmpfunc_increasing);  
  start = seconds();
  res = solve_knapsack_v3(vols_ordered_increasing, n_vols, capacity);
  end = seconds() - start;
  printf("v3 o CPU, res: %d, elapsed: %f\n", res, end * 1000);

  //let's try also with the ordered volumes, but in reverse
  int vols_ordered_decreasing[n_vols];
  for(int i = 0; i < n_vols; i++){
    vols_ordered_decreasing[i] = vols[i];
  }
  qsort(vols_ordered_decreasing, n_vols, sizeof(int), cmpfunc_decreasing);  
  start = seconds();
  res = solve_knapsack_v3(vols_ordered_decreasing, n_vols, capacity);
  end = seconds() - start;
  printf("v3 ro CPU, res: %d, elapsed: %f\n", res, end * 1000);


  start = seconds();
  res = solve_knapsack_v4(vols, n_vols, capacity);
  end = seconds() - start;
  printf("v4 CPU, res: %d, elapsed: %f\n", res, end * 1000);

  //let's try (again) with the ordered volumes to see if we have a speedup
  start = seconds();
  res = solve_knapsack_v4(vols_ordered_increasing, n_vols, capacity);
  end = seconds() - start;
  printf("v4 o CPU, res: %d, elapsed: %f\n", res, end * 1000);

  //let's try also with the ordered volumes, but in reverse
  start = seconds();
  res = solve_knapsack_v4(vols_ordered_decreasing, n_vols, capacity);
  end = seconds() - start;
  printf("v4 ro CPU, res: %d, elapsed: %f\n", res, end * 1000);



  //----------------------------------------------------------------------------
  //-------------------------------GPU ALGORITHMS-------------------------------
  //----------------------------------------------------------------------------

  dim3 block(1024);
  dim3 grid((capacity + 1) + block.x - 1);


  //first, we need to declare host and device memory, and initialize it
  
  int* row_h = (int*) malloc((capacity + 1) * sizeof(int));

  int *old_row_d, *new_row_d;
  cudaMalloc(&old_row_d, capacity + 1);
  cudaMalloc(&new_row_d, capacity + 1);

  start = seconds();
  cudaEvent_t eStart, eEnd;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eEnd);
  cudaEventRecord(eStart);

  //first step: create the initial row
  kernel_v1_a<<<grid, block>>>(vols[0], old_row_d, capacity);
  cudaDeviceSynchronize();
  cudaMemcpy(row_h, old_row_d, capacity + 1, cudaMemcpyDeviceToHost);

  //second step: create all the subsequent rows
  for(int r = 1; r < n_vols; r++){
    cudaMemcpy(old_row_d, row_h, capacity + 1, cudaMemcpyHostToDevice);
    kernel_v1_b<<<grid, block>>>(vols[r], old_row_d, new_row_d, capacity);
    cudaDeviceSynchronize();
    cudaMemcpy(row_h, new_row_d, capacity + 1, cudaMemcpyDeviceToHost);
  }

  cudaEventRecord(eEnd);
  cudaEventSynchronize(eEnd);

  float msEvent;
  cudaEventElapsedTime(&msEvent, eStart, eEnd);

  end = seconds() - start;
  printf("v1 GPU, res: %d, elapsed: %f, event time: %f\n", row_h[capacity], end * 1000, msEvent);



  //finally, release the memory

  cudaEventDestroy(eStart);
  cudaEventDestroy(eEnd);

  free(row_h);

  cudaFree(old_row_d);
  cudaFree(new_row_d);


  return 0;

}