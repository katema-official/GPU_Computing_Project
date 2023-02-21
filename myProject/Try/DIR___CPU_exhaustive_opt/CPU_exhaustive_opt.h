#ifndef CPU_exhaustive_opt
#define CPU_exhaustive_opt

int subsetSumOptimization_recursive(int* volumes, int capacity, int n_volumes);
int* subsetSumOptimization_recursive_solFound(int solution, int* volumes, int volume_occupied, int n_volumes, int idx_current_elem);
int subsetSumOptimization_recursive_copies(int* volumes, int* copies, int capacity, int n_volumes);

#endif