#include <stdlib.h>
#include <stdio.h>

#define N 10

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



    return 0;
}