#include <time.h>

inline double seconds(){
    return clock();
            
}

inline double elapsedTime(double start, double end){
    return ((double) (end-start)) / CLOCKS_PER_SEC;
}