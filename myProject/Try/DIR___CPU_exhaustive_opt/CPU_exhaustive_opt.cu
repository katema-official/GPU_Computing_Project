#include <stdio.h>

int subsetSumOptimization_recursive(int* volumes, int capacity, int n_volumes);
int* subsetSumOptimization_recursive_solFound(int solution, int* volumes, int volume_occupied, int n_volumes, int idx_current_elem);
int subsetSumOptimization_recursive_constructive(int* volumes, int capacity, int n_volumes);
int SSO_rec_const(int accum, int* volumes, int capacity, int n_volumes);
int subsetSumOptimization_recursive_destructive(int* volumes, int capacity, int n_volumes);
int SSO_rec_dest(int accum, int* volumes, int capacity, int n_volumes);

int subsetSumOptimization_recursive_copies(int* volumes, int* copies, int capacity, int n_volumes);
int subsetSumOptimization_recursive_constructive_copies(int* volumes, int* copies, int capacity, int n_volumes);
int SSO_rec_const_copies(int accum, int* volumes, int* copies, int capacity, int n_volumes);
int subsetSumOptimization_recursive_destructive_copies(int* volumes, int* copies, int capacity, int n_volumes);
int SSO_rec_dest_copies(int accum, int* volumes, int* copies, int capacity, int n_volumes);


//recursive approach with exhaustive search (for the subsetSumOptimization)
int subsetSumOptimization_recursive(int* volumes, int capacity, int n_volumes){
	int total_sum = 0;
	for(int i = 0; i < n_volumes; i++){
		total_sum += volumes[i];
	}
	
	float mean = total_sum / n_volumes;
	
	int res;
	
	//if we estimate that more than half of the volumes are necessary to fill the capacity (at its best), we use the destructive approach.
	//otherwise, we use the constructive one.
	if((capacity / mean) > (n_volumes / 2)){
		printf("destructive\n");
		res = subsetSumOptimization_recursive_destructive(volumes, capacity, n_volumes);
	}else{
		printf("constructive\n");
		res = subsetSumOptimization_recursive_constructive(volumes, capacity, n_volumes);
	}
	
	return res;
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





int subsetSumOptimization_recursive_constructive(int* volumes, int capacity, int n_volumes){
	return SSO_rec_const(0, volumes, capacity, n_volumes);
}

int SSO_rec_const(int accum, int* volumes, int capacity, int n_volumes){
	if(n_volumes == 0) return accum;
  
  if(accum + volumes[0] <= capacity){
    int a = SSO_rec_const(accum + volumes[0], &(volumes[1]), capacity, n_volumes - 1);
    int b = SSO_rec_const(accum, &(volumes[1]), capacity, n_volumes - 1);
    if(a >= b){
      return a;
    }else{
      return b;
    }
  }else{
    return SSO_rec_const(accum, &(volumes[1]), capacity, n_volumes - 1);
  }
	
}



int subsetSumOptimization_recursive_destructive(int* volumes, int capacity, int n_volumes){
	int accum = 0;
	for(int i = 0; i < n_volumes; i++){
		accum += volumes[i];
	}
	
	return SSO_rec_dest(accum, volumes, capacity, n_volumes);
}

int SSO_rec_dest(int accum, int* volumes, int capacity, int n_volumes){
	if(n_volumes == 0){
		if(accum > capacity) accum = 0;	//trash unfeasible solutions
		return accum;
	}
	if(accum <= capacity) return accum;
	
	int a = SSO_rec_dest(accum - volumes[0], &(volumes[1]), capacity, n_volumes - 1);
	int b = SSO_rec_dest(accum, &(volumes[1]), capacity, n_volumes - 1);	
	
	if(a >= b){
		return a;
	}else{
		return b;
	}
	
	
}



//----------------------------------------------------------------------------------------------------------------



//The version where two input arrays are given: one contains the volumes, the other their number of copies.
int subsetSumOptimization_recursive_copies(int* volumes, int* copies, int capacity, int n_volumes){
	int total_n = 0;
	for(int i = 0; i < n_volumes; i++){
			total_n += copies[i];
	}
	int total_sum = 0;
	for(int i = 0; i < n_volumes; i++){
		total_sum += copies[i]*volumes[i];
	}
	
	float mean = total_sum / total_n;
	
	int res;
	
	//if we estimate that more than half of the volumes are necessary to fill the capacity (at its best), we use the destructive approach.
	//otherwise, we use the constructive one.
	if((capacity / mean) > (total_n / 2)){
		res = subsetSumOptimization_recursive_destructive_copies(volumes, copies, capacity, n_volumes);
	}else{
		res = subsetSumOptimization_recursive_constructive_copies(volumes, copies, capacity, n_volumes);
	}
	
	return res;
}





//This algorithm is particularly efficient when it is known a priori that the number of items that make the optimal solution
//is less or equal than (total number of volumes) / 2, where (total number of volumes) = sum of all the values in copies.
int subsetSumOptimization_recursive_constructive_copies(int* volumes, int* copies, int capacity, int n_volumes){
	int* copy = (int*) malloc(n_volumes * sizeof(int));
	for(int i = 0; i < n_volumes; i++){
		copy[i] = copies[i];
	}
	int res = SSO_rec_const_copies(0, volumes, copy, capacity, n_volumes);
	
	free(copy);
	
	return res;
}

int SSO_rec_const_copies(int accum, int* volumes, int* copies, int capacity, int n_volumes){
	if(n_volumes == 0) return accum;
	
	if(copies[0] == 1){
		if(accum + volumes[0] <= capacity){
			int a = SSO_rec_const_copies(accum + volumes[0], &(volumes[1]), &(copies[1]), capacity, n_volumes - 1);
			int b = SSO_rec_const_copies(accum, &(volumes[1]), &(copies[1]), capacity, n_volumes - 1);
			if(a >= b){
				return a;
			}else{
				return b;
			}
		}else{
			return SSO_rec_const_copies(accum, &(volumes[1]), &(copies[1]), capacity, n_volumes - 1);
		}
	}else{
		copies[0] = copies[0] - 1;
		if(accum + volumes[0] <= capacity){
			int a = SSO_rec_const_copies(accum + volumes[0], volumes, copies, capacity, n_volumes);
			int b = SSO_rec_const_copies(accum, volumes, copies, capacity, n_volumes);
			if(a >= b){
				return a;
			}else{
				return b;
			}
		}else{
			return SSO_rec_const_copies(accum, volumes, copies, capacity, n_volumes);
		}
	}
}





//This algorithm is particularly efficient when it is known a priori that the number of items that make the optimal solution
//is greater or equal than (total number of volumes) / 2, where (total number of volumes) = sum of all the values in copies.
int subsetSumOptimization_recursive_destructive_copies(int* volumes, int* copies, int capacity, int n_volumes){
	int* copy = (int*) malloc(n_volumes * sizeof(int));
	for(int i = 0; i < n_volumes; i++){
		copy[i] = copies[i];
	}
	int accum = 0;
	for(int i = 0; i < n_volumes; i++){
		accum += copy[i]*volumes[i];
	}
	int res = SSO_rec_dest_copies(accum, volumes, copy, capacity, n_volumes);
	
	free(copy);
	
	return res;
}

int SSO_rec_dest_copies(int accum, int* volumes, int* copies, int capacity, int n_volumes){
	if(n_volumes == 0){
		if(accum > capacity) accum = 0;	//trash unfeasible solutions
		return accum;
	}
	if(accum <= capacity) return accum;
	
	int a, b;
	if(copies[0] == 1){
		a = SSO_rec_dest_copies(accum - volumes[0], &(volumes[1]), &(copies[1]), capacity, n_volumes - 1);
		b = SSO_rec_dest_copies(accum, &(volumes[1]), &(copies[1]), capacity, n_volumes - 1);
	}else{
		copies[0] = copies[0] - 1;
		a = SSO_rec_dest_copies(accum - volumes[0], volumes, copies, capacity, n_volumes);
		b = SSO_rec_dest_copies(accum, volumes, copies, capacity, n_volumes);
	}
	
	if(a >= b){
		return a;
	}else{
		return b;
	}
}
