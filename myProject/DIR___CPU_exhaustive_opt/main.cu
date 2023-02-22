#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "CPU_exhaustive_opt.h"
#include "../common.h"

//to run as:
// ./main [random_or_not] [n_vols] [capacity] [random_seed]
int main(int argc, char **argv){
    input_data data = initialize_1(argc, argv);

    int* volumes = data.volumes;
    int capacity = data.capacity;
    int n_vols = data.n_volumes;

    double start, end;

    //note that this is the worst strategy, but surely exact: exhaustive search
    start = seconds();
    int result_exhaustive = subsetSumOptimization_recursive(volumes, capacity, n_vols);
    end = seconds();
    printf("recursive exhaustive CPU optimization, res: %d, elapsed: %f\n", result_exhaustive, elapsedTime(start, end));

    //if one has enough time, with this call it can see the actual solution (the optimal subset)
    if(PRINT_SUBSET_SOL){
        int* subset;
        subset = subsetSumOptimization_recursive_solFound(result_exhaustive, volumes, 0, n_vols, 0);
        for(int i = 0; i < n_vols; i++){
            if(subset[i] == 1){
            printf("%d + ", volumes[i]);
            }
        }
        printf(" = %d\n", result_exhaustive);
    }

}