//recursive approach with exhaustive search (for the subsetSumOptimization)
int subsetSumOptimization_recursive(int accum, int* volumes, int capacity, int n_volumes){
  if(n_volumes == 0) return accum;
  
  if(accum + volumes[0] <= capacity){
    int a = subsetSumOptimization_recursive(accum + volumes[0], &(volumes[1]), capacity, n_volumes - 1);
    int b = subsetSumOptimization_recursive(accum, &(volumes[1]), capacity, n_volumes - 1);
    if(a >= b){
      return a;
    }else{
      return b;
    }
  }else{
    return subsetSumOptimization_recursive(accum, &(volumes[1]), capacity, n_volumes - 1);
  }
}

//Proof of correctness: this algorithm takes the volumes, the capacity, the number of volumes and the solution
//found by some other algorithm, and returns the array of elements that sum up to that solution
int* subsetSumOptimization_recursive_solFound(int solution, int* volumes, int volume_occupied, int n_volumes, int idx_current_elem){
  if(volume_occupied > solution) return NULL;

  if(volume_occupied == solution){
    int* sol = (int*) malloc(n_volumes * sizeof(int));
    for(int i = 0; i < n_volumes; i++){
      sol[i] = 0;
    }
    return sol;
  }

  if(n_volumes == idx_current_elem) return NULL;

  int* a_sol = subsetSumOptimization_recursive_solFound(solution, volumes, volume_occupied + volumes[idx_current_elem], n_volumes, idx_current_elem + 1);
  int* b_sol = subsetSumOptimization_recursive_solFound(solution, volumes, volume_occupied, n_volumes, idx_current_elem + 1);

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