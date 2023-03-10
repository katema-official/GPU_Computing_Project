#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "common.h"

#include "CPU_exhaustive_opt.h"
#include "CPU_exhaustive_dec.h"
#include "CPU_dp.h"
#include "GPU_dp_basic.h"
#include "GPU_dp_smem.h"
#include "GPU_dp_uroll.h"
#include "GPU_dp_uroll8_smem.h"



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


