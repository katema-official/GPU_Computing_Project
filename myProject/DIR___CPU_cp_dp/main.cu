#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../common.h"
#include "CPU_dp.h"

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

    

    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;
    for(int i = 0; i <= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(subsetSumDecision_DP_v1(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds();
    printf("DP v1 CPU, res: %d, elapsed: %f\n", res, elapsedTime(start, end));
  


    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;
    for(int i = 0; i<= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(subsetSumDecision_DP_v2(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds();
    printf("DP v2 CPU, res: %d, elapsed: %f\n", res, elapsedTime(start, end));
  


    start = seconds();
    res = capacity;
    n_vols_agumented = n_vols + (int) log2(capacity) + 1;
    for(int i = 0; i<= (int) log2(capacity); i++){
      n_vols_agumented--;
      if(subsetSumDecision_DP_v3(vols_agumented, res, n_vols_agumented) == FALSE){
        res = res - vols_agumented[n_vols_agumented];
      }
    }
    end = seconds();
    printf("DP v3 CPU, res: %d, elapsed: %f\n", res, elapsedTime(start, end));
  



}