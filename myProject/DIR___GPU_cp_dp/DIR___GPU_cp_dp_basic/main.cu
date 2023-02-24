#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../../common.h"
#include "GPU_dp_basic.h"

//to run as:
// ./main [initialize_type] [random_or_not] [n_vols] [capacity] [random_seed]
int main(int argc, char **argv){
    input_data data;
    
    switch(atoi(argv[1])){
        case 0:
            data = initialize_1(argc, argv);
            break;
        case 1:
            data = initialize_custom_1();
            break;
        case 2:
            data = initialize_custom_2();
            break;
        default:
            break;
    }

    int* volumes = data.volumes;
    int capacity = data.capacity;
    int n_vols = data.n_volumes;

    double start, end;

    //Credits for the original Subset Sum Optimization algorithm to Chad Parry:
    //https://courses.cs.washington.edu/courses/csep521/05sp/lectures/sss.pdf

    //first of all, following the algorithm, we have to agument the set of volumes with all the
    //powers of 2 smaller than capacity
    int* vols_agumented = (int*) malloc((n_vols + log2(capacity) + 1) * sizeof(int));
    for(int i = 0; i < n_vols; i++){
        vols_agumented[i] = volumes[i];
    }

    for(int i = log2(capacity); i >= 0; i--){
        int add = pow(2, i);
        vols_agumented[n_vols + i] = add;
    }
    
    int res;
    int n_vols_agumented;

    
    //Since here no shared memory is used...
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
    if(0){
    //This approach (v1) is incredibly slow because of memory copies between host and device
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

    end = seconds();
    msEvent = msEvent / 1000;
    printf("DP v1 GPU, res: %d, elapsed: %f, event time: %f\n", res, elapsedTime(start, end), msEvent);

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

    end = seconds();
    msEvent = msEvent / 1000;
    printf("DP v2 GPU, res: %d, elapsed: %f, event time: %f\n", res, elapsedTime(start, end), msEvent);

    CHECK(cudaEventDestroy(eStart));
    CHECK(cudaEventDestroy(eEnd));




    //finally, release the memory
    free(row_h);

    CHECK(cudaFree(old_row_d));
    CHECK(cudaFree(new_row_d));
    CHECK(cudaDeviceReset());

    free(vols_agumented);
    free(volumes);
  



}