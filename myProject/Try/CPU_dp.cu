#include "common.h"
#include "stdio.h"

//Dynamic programming approach

//v1: build a dynamic programming matrix of n_items+1 rows and capacity+1 columns.
//Inefficient from memory point of view
//Inefficient because some operations are perfermed even when not necessary

unsigned char subsetSumDecision_DP_v1(int* volumes, int capacity, int n_items){
    unsigned char** B = (unsigned char**) malloc((n_items+1)*sizeof(unsigned char*));
    for(int i = 0; i < n_items+1; i++){
        B[i] = (unsigned char*) malloc((capacity+1)*sizeof(unsigned char));
        if(B[i] == NULL){
            printf("Allocation failed v1\n");
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


