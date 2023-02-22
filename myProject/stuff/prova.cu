#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../common.h"

//from now on, this is just old code

__global__ void kernel1(int nData){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nData){
    printf("Hello world from GPU!!! Thread number %d\n", idx);
  }
}

int main(int argc, char **argv) {
  printf("Hello world!\n");
  int nData = 10;
  if(argc>1){
    nData = atoi(argv[1]);
  }
  dim3 block(10);
  dim3 grid((nData + block.x - 1)/block.x);
  kernel1<<<grid, block>>>(nData);
  cudaDeviceSynchronize();
  return 0;
}