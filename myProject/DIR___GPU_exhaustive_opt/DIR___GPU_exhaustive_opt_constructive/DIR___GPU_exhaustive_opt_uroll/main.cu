#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../../../common.h"
#include "GPU_exhaustive_opt_uroll.h"






//to run as:
// ./main [initialize_type] [random_or_not] [n_vols] [capacity] [random_seed] [jump]
//Note that n_vols MUST correspond to the N marco value
//Moreover, jump must always be a power of 2
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
    int n_volumes = data.n_volumes;

    printf("N = %d, n = %d\n", N, n_volumes);
    assert(N == n_volumes);

    int jump = 1024;
    if(argc > 6){
        jump = atoi(argv[6]);
    }

    //Since here no shared memory is used...
    CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    //double start, end;
    cudaEvent_t eStart, eEnd;
    float msEvent;

    //start = seconds();

    CHECK(cudaEventCreate(&eStart));
    CHECK(cudaEventCreate(&eEnd));
    CHECK(cudaEventRecord(eStart, 0));

    int res = subsetSumOptimization_exhaustive_GPU(volumes, capacity, jump);

    CHECK(cudaEventRecord(eEnd, 0));
    CHECK(cudaEventSynchronize(eEnd));

    CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

    //end = seconds();
    msEvent = msEvent / 1000;
    printf("recursive exhaustive GPU optimization uroll, res: %d, event time: %f\n", res, msEvent);

    CHECK(cudaEventDestroy(eStart));
    CHECK(cudaEventDestroy(eEnd));

    free(volumes);
	return 0;
}