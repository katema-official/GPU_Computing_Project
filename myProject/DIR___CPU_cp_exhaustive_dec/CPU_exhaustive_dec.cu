#include "../common.h"

//recursive approach with exhaustive search (for the subsetSumDecision, it will be useful in Chad Parry's algorithm)
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
    int* sol = (int*) malloc(n_volumes * sizeof(int));
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