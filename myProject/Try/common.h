#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>


#define DEBUG_1 0
#define TRUE 1
#define FALSE 0
#define PRINT_SUBSET_SOL 0

#define BLOCK_DIM_X 128

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


//Function to be called by a main to initialize a bit the data of the problem (better procedres could be generated)
//arguments for argc: [random_or_not] [n_vols] [capacity] [random_seed]
//returns the array of volumes
inline input_data initialize_1(int argc, char** argv){
    int n_vols = 32;
    int* vols;
    int capacity = 10000;

    input_data result;

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
    if(DEBUG_1 || 1){
        for(int i = 0; i < n_vols; i++){
            printf("vols[%d] = %d\n", i, vols[i]);
        }
    }



    result.volumes = vols;
    result.capacity = capacity;
    result.n_volumes = n_vols;

    return result;
}


#endif // _COMMON_H