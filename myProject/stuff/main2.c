#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <time.h>

double seconds(){
    return clock(); 
}

double elapsedTime(double start, double end){
    return ((double) (end-start)) / CLOCKS_PER_SEC;
}

#define DEBUG_1 0
#define TRUE 1
#define FALSE 0

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------CPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//recursive approach with exhaustive search
int subsetSumDecision_recursive(int current_idx, int* volumes, int capacity, int n_volumes){
  if(capacity == 0) return TRUE;
  if(n_volumes == current_idx) return FALSE;

    printf("current_idx = %d\n", current_idx);

  return 
    subsetSumDecision_recursive(current_idx + 1, volumes, capacity - volumes[current_idx], n_volumes) ||
    subsetSumDecision_recursive(current_idx + 1, volumes, capacity, n_volumes);

}

//just if someone wants to be sure, here is a version that also gives the elements of the solution
//NOT NULL in solution = this is the solution that solves the problem with TRUE
//NULL in solution: FALSE
int* subsetSumDecision_recursive_solFound(int* volumes, int capacity, int n_volumes, int idx_current_elem){
  if(capacity == 0){
    int* sol = malloc(n_volumes * sizeof(int));
    for(int i = 0; i < n_volumes; i++){
      sol[i] = 0;
    }
    return sol;
  }

  if(n_volumes == idx_current_elem) return NULL;

  int* a_sol = subsetSumDecision_recursive_solFound(volumes, capacity - volumes[idx_current_elem], n_volumes, idx_current_elem + 1);
  int* b_sol = subsetSumDecision_recursive_solFound(volumes, capacity, n_volumes, idx_current_elem + 1);

  if((a_sol == NULL) && (b_sol == NULL)){
    return NULL;
  }

  if(a_sol != NULL){
    a_sol[idx_current_elem] = 1;
    return a_sol;
  }

  if(b_sol != NULL){
    b_sol[idx_current_elem] = 0;
    return b_sol;
  }

  return NULL;
}





//Dynamic programming approach

//v1: build a dynamic programming matrix of n_items+1 rows and capacity+1 columns.
//Inefficient from memory point of view
//Inefficient because some operations are perfermed even when not necessary

unsigned char subsetSumDecision_DP_v1(int* volumes, int capacity, int n_items){
    unsigned char** B = (unsigned char**) malloc((n_items+1)*sizeof(unsigned char*));
    if(B!=NULL) printf("A A A\n");
    for(int i = 0; i < n_items+1; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
            printf("Allocation failed v1\n");
        }else{
            printf("OOOOK\n");
        }
    }
    
    //initialization: the subproblems without items and capacity 0 admit a solution
    for(int i = 0; i < n_items+1; i++){
        B[i][0] = TRUE;
    }
    //initialization: the subproblems without items but a bit of capacity don't admit a solution
    for(int i = 1; i < capacity+1; i++){
        B[0][i] = FALSE;
    }

    unsigned char res = 0;

    //now, the value of each cell of each row can be fully determined by the the previous row
    for(int row = 1; row < n_items + 1; row++){
        int volume_row = volumes[row-1];
        for(int col = 1; col < capacity + 1; col++){
            if(col >= volume_row){
              B[row][col] = B[row - 1][col] || B[row - 1][col - volume_row];
            }else{
              B[row][col] = B[row - 1][col];  //copy the previous entry
            }
        }

        if(DEBUG_1) printf("temporary result: %d\n", B[row][capacity]);

        if(B[row][capacity] == 1){
            res = B[row][capacity];
            break;
        }
    }

    for(int i = 0; i < n_items+1; i++){
        free(B[i]);
    }
    free(B);

    return res;
}

//v2: same as before, but using only 2 rows to use less memory. There is however the added complexity of copying the new row in the old one.
//Inefficient because of multiple memory copies
//Still inefficient because some operations are performed even when not necessary

unsigned char subsetSumDecision_DP_v2(int* volumes, int capacity, int n_items){
    unsigned char** B = (unsigned char**) malloc(2*sizeof(unsigned char*));
    for(int i = 0; i < 2; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
          printf("Allocation failed v2\n");
        }
    }
    
    for(int i = 0; i < 2; i++){
        B[i][0] = TRUE;
    }
    for(int i = 1; i < capacity+1; i++){
        B[0][i] = FALSE;
    }

    unsigned char res = 0;

    //now, the value of each cell of each row can be fully determined by the the previous row
    for(int iteration = 0; iteration < n_items; iteration++){
        int volume_row = volumes[iteration];
        for(int col = 1; col < capacity + 1; col++){
            if(col >= volume_row){  //this item could be part of the solution
                B[1][col] = B[0][col] || B[0][col - volume_row];
            }else{
                B[1][col] = B[0][col];  //the volume of this item is more than the current capacity
            }
        }

        //now copy the new row in the old one
        for(int col = 1; col < capacity + 1; col++){
            B[0][col] = B[1][col];
        }

        if(DEBUG_1) printf("temporary result: %d\n", B[0][capacity]);

        if(B[0][capacity] == 1){
            res = B[1][capacity];
            break;
        }

    }

    for(int i = 0; i < 2; i++){
        free(B[i]);
    }
    free(B);

    return res;
}

//v3: doing everything in one row. There is no overhead because of copy operations.
//We also avoid performing useless operations.

unsigned char subsetSumDecision_DP_v3(int* volumes, int capacity, int n_items){
    if(capacity == 0 || n_items == 0){
        return 0;
    }
    unsigned char* B = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
    for(int i = capacity; i > 0; i--){
        B[i] = FALSE;
    }
    B[0] = TRUE;

    unsigned char res = 0;

    //now, the value of each cell of each row can be fully determined by the the previous row,
    //that is actually the same row
    for(int iteration = 0; iteration < n_items; iteration++){
        int volume_row = volumes[iteration];
        for(int col = capacity; col >=volume_row; col--){
            B[col] = B[col] || B[col - volume_row];
            //printf("B[%d] = %d ", col, B[col]);
        }

        if(DEBUG_1) printf("temporary result: %d\n", B[capacity]);

        if(B[capacity] == 1){
            res = B[capacity];
            break;
        }
    }

    free(B);
    return res;
}




int cmpfunc_increasing(const void * a, const void * b) {
   return (*(int*)a - *(int*)b);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------GPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//v1: Compute on the device one row, than bring it back to the host.
//then copy it again on the device and use it to compute the new row.
//-Inefficient because of multiple copies between device and host

/*
__global__ void kernel_v1_a(int v, unsigned char* res_row, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx == 0){
    res_row[idx] = TRUE;
  }else{
    if(idx <= capacity){
        res_row[idx] = FALSE;
    }
  }
}

__global__ void kernel_v1_b(int v, unsigned char* input_row, unsigned char* output_row, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < v){
    output_row[idx] = input_row[idx];
  }else{
    if(idx <= capacity) output_row[idx] = input_row[idx] || input_row[idx - v];
  }
}



//v2: do everything in one kernel, minimizing the copies in global memory
//and doing only the strictly necessary data transfers between device and host

__global__ void kernel_v2_a(int init_v, unsigned char* row_0, int capacity){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx == 0){
    row_0[idx] = TRUE;
  }else{
    if(idx <= capacity){
      row_0[idx] = FALSE;
    }
  }
}

__global__ void kernel_v2_b(int v, int* row_0, int* row_1, int capacity, int old_row_idx){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //now, take the last row that has been computed, and use it to compute the new
  //one, that will be the opposite row. This avoids us to copy data from row_1
  //to row_0 for o(vols) times.
  
  if(old_row_idx == 0){

    //compute row_1 from row_0
    if(idx < v){
      row_1[idx] = row_0[idx];
    }else{
      if(idx <= capacity) row_1[idx] = row_0[idx] || row_0[idx - v];
    }

  }else{

    //compute row_0 from row_1
    if(idx < v){
      row_0[idx] = row_1[idx];
    }else{
      if(idx <= capacity) row_0[idx] = row_1[idx] || row_1[idx - v];
    }

  }

}

*/










//to run as:
// ./main [random_or_not] [n_vols] [capacity] [random_seed]

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

  //first of all, following the algorithm, we have to agument the set of volumes with all the
  //powers of 2 smaller than capacity

  int n_vols_agumented = n_vols + (int) log2(capacity) + 1;
  int* vols_agumented = (int*) malloc((n_vols_agumented) * sizeof(int));
  //memcpy(&vols_agumented, &vols, n_vols*sizeof(int));
  printf("n_vols = %d, log2(capacity) = %d, n_vols_agumented = %d\n", n_vols, (int) log2(capacity), n_vols_agumented);

  for(int i = 0; i < n_vols; i++){
    vols_agumented[i] = vols[i];
  }

  int log = (int) log2(capacity);
  for(int i = log; i >= 0; i--){
    printf("a vols_agumented[%d] = ?, ", n_vols + i);
    int add = pow(2, i);
    vols_agumented[n_vols + i] = add;
    printf("b vols_agumented[%d] = %d, ", n_vols + i, vols_agumented[n_vols + i]);
  }

  start = seconds();
  int res = capacity;
  for(int i = 0; i <= (int) log2(capacity); i++){
    n_vols_agumented--;   //simulates A <- A \ 2^i
    if(subsetSumDecision_recursive(0, vols_agumented, res, n_vols_agumented) == FALSE){
      res = res - vols_agumented[n_vols_agumented];
    }
  }
  end = seconds() - start;
  printf("recursive exhaustive CPU, res: %d, elapsed: %f\n", res, end * 1000);

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
  printf("DP v1 CPU, res: %d, elapsed: %f\n", res, end * 1000);


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
  printf("DP v2 CPU, res: %d, elapsed: %f\n", res, end * 1000);


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
  printf("DP v3 CPU, res: %d, elapsed: %f\n", res, end * 1000);







  free(vols_agumented);
  free(vols);


}


