#include <stdlib.h>
#include <stdio.h>

#define N 10


typedef struct{
    int* s_array;
}my_struct;

my_struct func_struct(my_struct* ms){
    //int a[36] = {0, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    //20, 21, 22, 23,24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
    my_struct ret;
    ret.s_array = (int*) malloc(5*sizeof(int));

    int a[5]= {9, 10, 11, 12, 13};

    for(int i = 0; i < 5; i++){
        ret.s_array[i] = a[i];
    }

    

    /*ret.s_array[0] = 10;
    ret.s_array[1] = 11;
    ret.s_array[2] = 12;
    ret.s_array[3] = 13;
    ret.s_array[4] = 15;*/

    return ret;
    
    //ms->s_array = a;
    
}



void func(int array[N]){
    for(int i = 0; i < N; i++){
        array[i] = i;
    }
}

int main(){
    int my_array[N];
    for(int i = 0; i < N; i++){
        my_array[i] = 0;
    }
    for(int i = 0; i < N; i++){
        printf("%d ", my_array[i]);
    }
    printf("\n");

    func(my_array);

    for(int i = 0; i < N; i++){
        printf("%d ", my_array[i]);
    }


    char pippo[80];
    for(int i = 0; i < 80; i++){
        pippo[i] = '0';
    }
    for(int i = 0; i < 80; i++){
        printf("%c",pippo[i]);
    }
    printf("\n");

    //my_struct* struct_; // = (my_struct*) malloc(sizeof(my_struct));
    my_struct struct_;
    int n = 5;
    //struct_.s_array = (int*) malloc(n*sizeof(int));

    my_struct return_struct = func_struct(&struct_);

    for(int i = 0; i < 5; i++){
        printf("%d ", return_struct.s_array[i]);
    }

    int* ret;
    ret = return_struct.s_array;
    printf("\n");
    for(int i = 0; i < n; i++){
        printf("%d ", ret[i]);
    }

    return 0;
}