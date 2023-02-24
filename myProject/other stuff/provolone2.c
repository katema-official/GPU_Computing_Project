#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4

//Second, we need the code that sums two bit strings of the same length.
//This assumes that the two strings have the same length N.
//The result is stored in the first string.
//Adapted from: https://www.geeksforgeeks.org/add-two-bit-strings/
void add_bit_strings(char str1[N], char str2[N]){
	//Initialize carry
	int carry = (int) '0'; 

	//Add all bits one by one
	for (int i = N - 1; i >= 0; i--)
	{
		int firstBit = (int) str1[i];
		int secondBit = (int) str2[i];

		//Boolean expression for sum of 3 bits
		int sum = (firstBit ^ secondBit ^ carry);

		str1[i] = (char) sum;

		// boolean expression for 3-bit addition
		carry = (firstBit & secondBit) |
								(secondBit & carry) |
								(firstBit & carry);
	}

	// if overflow, then add a leading 1
	if (carry == 1)
	{
		printf("overflow!");
		//result = "1" + result;
	}
}

//Third, a function that translates an integer into its bit string representation.
//The char array in which the binary representation will be conserved is passed as argument
void convert_to_binary(char res[N], int number){
	int k;
	int l = (int) log2f(number) + 1;
	int i = N - l;

    char chachacha;
	for(int c = l - 1; c >= 0; c--){
		k = number >> c;
        res[i] = ((char) k&1) + '0';

		/*if (k & 1){
			res[i] = '1';
		}else{
			res[i] = '0';
		}*/
		i++;
	}
	for(int c = 0; c < N - l; c++){
		res[c] = '0';
	}
}


int main(){

    char binary_string_1[N];
    int number = 14;
    convert_to_binary(binary_string_1, number);
    for(int i = 0; i < N; i++){
        printf("%c", binary_string_1[i]);
    }



}


