#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../../../common.h"
#include "GPU_exhaustive_opt_basic.h"






//to run as:
// ./main [random_or_not] [n_vols] [capacity] [random_seed] [jump]
//Note that n_vols MUST correspond to the N marco value
//Moreover, jump must always be a power of 2
int main(int argc, char **argv){
    input_data data = initialize_1(argc, argv);

    int* volumes = data.volumes;
    int capacity = data.capacity;

    int jump = 1024;
    if(argc > 5){
        jump = atoi(argv[5]);
    }

    //Since here no shared memory is used...
    CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    double start, end;
    cudaEvent_t eStart, eEnd;
    float msEvent;

    start = seconds();

    CHECK(cudaEventCreate(&eStart));
    CHECK(cudaEventCreate(&eEnd));
    CHECK(cudaEventRecord(eStart, 0));

    int res = subsetSumOptimization_exhaustive_GPU(volumes, capacity, jump);

    CHECK(cudaEventRecord(eEnd, 0));
    CHECK(cudaEventSynchronize(eEnd));

    CHECK(cudaEventElapsedTime(&msEvent, eStart, eEnd));

    end = seconds();
    msEvent = msEvent / 1000;
    printf("recursive exhaustive GPU optimization, res: %d, elapsed: %f, event time: %f\n", res, elapsedTime(start, end), msEvent);

    CHECK(cudaEventDestroy(eStart));
    CHECK(cudaEventDestroy(eEnd));

    free(volumes);
	return 0;
}