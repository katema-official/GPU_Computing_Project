#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../common.h"

#define DEBUG_1 1
#define TRUE 1
#define FALSE 0

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-----------------------------------CPU ZONE-----------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//recursive approach with exhaustive search
int subsetSumDecision_recursive(int* volumes, int capacity, int n_volumes){
  if(capacity == 0) return TRUE;
  if(n_volumes == 0) return FALSE;

  return 
    subsetSumDecision_recursive(&(volumes[1]), capacity - volumes[0], n_volumes-1) ||
    subsetSumDecision_recursive(&(volumes[1]), capacity, n_volumes-1);

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
}





//Dynamic programming approach

//v1: build a dynamic programming matrix of n_items+1 rows and capacity+1 columns.
//Inefficient from memory point of view
//Inefficient because some operations are perfermed even when not necessary

unsigned char subsetSumDecision_DP_v1(int* volumes, int n_items, int capacity){
    unsigned char** B = (unsigned char**) malloc((n_items+1)*sizeof(unsigned char*));
    for(int i = 0; i < n_items+1; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
            printf("Allocation failed\n");
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

unsigned char subsetSumDecision_DP_v2(int* volumes, int n_items, int capacity){
    unsigned char** B = (unsigned char**) malloc(2*sizeof(unsigned char*));
    for(int i = 0; i < 2; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
          printf("Allocation failed\n");
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

unsigned char subsetSumDecision_DP_v3(int* volumes, int n_items, int capacity){
    if(capacity == 0 || n_items == 0){
        return 0;
    }
    unsigned char* B = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
    for(int i = capacity; i >0; i--){
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


//to run as:
// ./main [random_or_not] [n_vols] [capacity] [random_seed]

int main(int argc, char **argv){

  int prova[4] = {6,3,5,2};//{2,3,5,6}; 
  int cap = 14;
  int res = subsetSumDecision_recursive(prova, cap, 4);
  printf("RES: %d\n", res);

  int* sol = subsetSumDecision_recursive_solFound(prova, cap, 4, 0);
  if(sol != NULL){
    printf("SOL: ");
    for(int i = 0; i < 4; i++){
      if(sol[i] == 1){
        printf("%d, ", prova[i]);
      }
    }
    printf("\n");
    free(sol);
  }

  unsigned char res_byte = subsetSumDecision_DP_v1(prova, 4, cap);
  printf("RES AGAIN: %d\n", res_byte);

  res_byte = subsetSumDecision_DP_v2(prova, 4, cap);
  printf("RES AGAIN: %d\n", res_byte);

  res_byte = subsetSumDecision_DP_v3(prova, 4, cap);
  printf("RES AGAIN: %d\n", res_byte);

  int log;
  log = log2(16);
  printf("log2(16) = %d", log);
  

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

  //actually, reasoning about it, the array of volumes must be ordered from
  //lower volume to higher volume, otherwise some solutions might be lost
  qsort(vols, n_vols, sizeof(int), cmpfunc_increasing);

  //check the volumes
  if(DEBUG_1){
    for(int i = 0; i < n_vols; i++){
      printf("vols[%d] = %d\n", i, vols[i]);
    }
  }

  
  //----------------------------------------------------------------------------
  //-------------------------------CPU ALGORITHMS-------------------------------
  //----------------------------------------------------------------------------
  
  
  double start, end;










}
