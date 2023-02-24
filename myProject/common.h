#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>


#define DEBUG_1 0
#define DEBUG_ALL_FEASIBLE_SOLUTIONS 0
#define TRUE 1
#define FALSE 0
#define PRINT_SUBSET_SOL 0

#define BLOCK_DIM_X 1024

#define ON_MY_PC 1
#define EXECUTE_GPU_INEFFICIENT 0





#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

/*
inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline void device_name() {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}
*/

inline double seconds(){
    return clock();
            
}

inline double elapsedTime(double start, double end){
    return ((double) (end-start)) / CLOCKS_PER_SEC;
}










typedef struct{
    int* volumes;
    int capacity;
    int n_volumes;
}input_data;


//Function to be called by a main to initialize a bit the data of the problem (better procedures could be generated)
//arguments for argc: [-] [random_or_not] [n_vols] [capacity] [random_seed]
//returns the array of volumes
inline input_data initialize_1(int argc, char** argv){
    int n_vols = 32;
    int* vols;
    int capacity = 10000;

    input_data result;

    //the first arguments (starting from 2) tells if the sequence of volumes must be randomly generated (1)
    //or not (0)
    int generate_randomly_flag = 0;
    if(argc > 2){
        generate_randomly_flag = atoi(argv[2]);
    }

    //the second argument is the number of volumes. If 0, the default one is used.
    if(argc > 3){
        int _n_vols = atoi(argv[3]);
        if(_n_vols > 0){
            n_vols = _n_vols;
        }
    }
    vols = (int*) malloc(n_vols * sizeof(int));

    //the third argument is the total capacity. If 0, the default one is used.
    if(argc > 4){
        int _capacity = atoi(argv[4]);
        if(_capacity > 0){
            capacity = _capacity;
        }
    }

    //the fourth argument is the seed to be used in case of randomly generated volumes.
    //if 0, then the seed is randomized. Otherwise, the argument becomes the seed.
    if(generate_randomly_flag){
        int seed = 0;
        srand(time(0));
        if(argc > 5){
            seed = atoi(argv[5]);
            if(seed != 0){
                srand(seed);
            }
        }
        
        //"standard" values:
        //-lower = 50 ==> lower = capacity/200 
        //-upper = 500 ==> upper = capacity/20
        //-capacity = 10000;

        int lower = capacity/10000;
        int upper = capacity/100;   // /10;
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
    if(DEBUG_1 || 1){
        for(int i = 0; i < n_vols; i++){
            printf("vols[%d] = %d\n", i, vols[i]);
        }

        int sum = 0;
        for(int i = 0; i < n_vols; i++){
            sum += vols[i];
        }
        printf("the sum of ALL volumes is %d\n", sum);

    }



    result.volumes = vols;
    result.capacity = capacity;
    result.n_volumes = n_vols;

    return result;
}


//Just a different input function to declare custom input values.
inline input_data initialize_custom_1(){
    int n_volumes = 36;
    int vols[36] = {
        185000,
        185000,
        12000,
        12000,
        12000,
        12000,
        12000,
        12000,
        12000,
        12000,

        12000,
        12000,
        12000,
        12000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,

        3000,
        3000,
        3000,
        3000,
        180,
        180,
        180,
        180,
        180,
        180,

        125,
        125,
        125,
        125,
        40,
        40
        };
    int capacity = 369000 * 10;

    input_data result;

    result.volumes = (int*) malloc(n_volumes*sizeof(int));

    for(int i = 0; i < n_volumes; i++){
        result.volumes[i] = vols[i] * 10;
    }

    result.capacity = capacity;
    result.n_volumes = n_volumes;

    return result;
}


inline input_data initialize_custom_2(){
    /*int n_volumes = 36;
    int vols[36] = {
        8610420,
        12678120,
        3153601,
        13893560,
        13009122,
        9472000,
        15657250,
        28006125,
        28006125,
        9358560,

        10146627,
        4988625,
        12984399,
        2026024,
        2026024,
        1722240,
        2681775,
        22231000,
        13910520,
        7612800,

        7717892,
        8640000,
        18696600,
        1412400,
        1907400,
        1907400,
        14120000,
        11339000,
        15058659,
        5641650,

        27355420,
        9111313,
        8610420,
        12595500,
        9307919,
        15407700
        };
    int capacity = 162368727;*/

    int n_volumes = 100;
    int vols[100] = {
        960000, 960000,                                                                                                             //2
        130000, 130000, 130000, 130000, 130000, 130000, 130000, 130000,                                                             //8
        24000, 24000, 24000, 24000, 24000, 24000, 24000, 24000, 24000, 24000, 24000, 24000,                                         //12
        11000, 11000, 11000, 11000, 11000, 11000, 11000, 11000,                                                                     //8
        8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000,                                                     //12
        3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,                                                                 //10
        1400, 1400, 1400, 1400, 1400, 1400,                                                                                         //6
        190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190,                                                       //14
        170, 170, 170, 170, 170, 170, 170, 170,                                                                                     //8
        55, 55, 55, 55, 55, 55,                                                                                                     //6
        20, 20, 20, 20,                                                                                                             //4
        8, 8, 8, 8,                                                                                                                 //4
        3, 3, 3, 3, 3, 3                                                                                                            //6
        };

    int capacity = 2030529;     //optimal res: 2030529

    input_data result;

    result.volumes = (int*) malloc(n_volumes*sizeof(int));

    for(int i = 0; i < n_volumes; i++){
        result.volumes[i] = vols[i];
    }

    result.capacity = capacity;
    result.n_volumes = n_volumes;

    return result;
}













//-------------------------------------------------------

inline int comp_incr(const void * a, const void * b)
{
 return *(const int*)a - *(const int*)b;
}

//arguments:
//1) an accum for the current solution
//2) the volumes, the n_volumes and the capacity_max
//3) an array in which the results will be stored
//4) an index that contains the position in the result array to store the new result
inline void print_all_recursive(int accum, int* volumes, int n_volumes, int capacity_max, int **results, int* index_result){
  if(n_volumes == 0){
    //if((*index_result) % ((int) pow(2,28)) == 0) printf("i = %d\n", *index_result);

    if(accum >= capacity_max - 20000){
        int b = 0;
        for(int j = 0; j < *index_result; j++){
            if((*results)[j] == accum){
                b = 1;
                break;
            }
        }

        if(!b){
            (*results)[*index_result] = accum;
            *index_result += 1;
        }
    }
    return;
  }
  
  if(accum + volumes[0] <= capacity_max){
    print_all_recursive(accum + volumes[0], &(volumes[1]), n_volumes - 1, capacity_max, results, index_result);
  }
  print_all_recursive(accum, &(volumes[1]), n_volumes - 1, capacity_max, results, index_result);
}


//function that prints all the subset sums that are lower than capacity_max, in increasing order
inline void print_all(int* volumes, int n_volumes, int capacity_max){
    int* results = (int*) malloc(pow(2, 29) * sizeof(int));

    int i = 0;
    print_all_recursive(0, volumes, n_volumes, capacity_max, &results, &i);

    qsort(results, i, sizeof(int), comp_incr);

    for(int j = 0; j < i; j++){
        printf("res[%d] = %d\n", j, results[j]);
    }
}











#endif // _COMMON_H