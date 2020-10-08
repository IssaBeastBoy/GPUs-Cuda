
#include "stdio.h"
#include "stdlib.h"
#include <time.h>    // change this to use time.h for Microsoft
#include "math.h"


/*
	CPU input parameters	100 100		 -> 0.085 seconds
							100 200		 -> 0.172 seconds
							300 200		 -> 5.058 seconds
							400 300		 -> 17.943 seconds
							500 400		 -> 46.861 seconds
							500 500      -> 58.562 seconds
							700 400      -> 129.240 seconds
							800 800		 -> 390.25 seconds
							1000 1000	 -> 1052.600 seconds 

	GPU input parameters	100 100		 -> 0.027 seconds
							200 100		 -> 0.063 seconds
							300 200		 -> 0.332 seconds
							400 300		 -> 0.981 seconds
							500 400		 -> 2.410 seconds
							500 400		 -> 3.004 seconds 
							700 400      -> 6.190 seconds
							800 800		 -> 18.550 seconds
							1000 1000	 -> 46.476 seconds

	
	Compiler generates all the appropriate data Copyin, Copyout for the data management
	and allocates the data to the GPU and from the GPU to the CPU therefore no need for
	any type of copying of data to or from the device/host.

	Kernels are used for nested loop to ensure that they are parallelized to 
	improve execution time and only applied to the fillMatrix and MatrixMult
	function because copyMatrix and MakeMatrix functions will be to be done 
	in a particular order because they generate	dependencies which tends to 
	cause parallelization problems. 
	
	Due to the MatrixMult function generating large values, and storing them to array A for 
	even numbers and B for odd ones causing the registers/local memory to be spilled as the 
	values of arrays A and B get larger due to this function, two temp array are added	where 
	copies of array A and B are to be stored. These temp arrays are use to copy back the 
	original values of A and B not the mixture of the original and multiples ones from array C 
	allowing the showMatrix function to return actual values.
*/
int main (int argc, char **argv);

void fillMatrix(int size, float **restrict A) {	
	#pragma acc kernels
   for (int i = 0; i < size; ++i) {	   
      for (int j = 0; j < size; ++j) {
        A[i][j] = ((float)i);
      }
   }
}
float** MatrixMult(int size, float **restrict A, float **restrict B, float **restrict C) {
		#pragma acc kernels
		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				float tmp = 0.;
				#pragma acc loop reduction(+:tmp)
				for (int k = 0; k < size; ++k) {
					tmp += A[i][k] * B[k][j];
				}
				C[i][j] = tmp;
			}
		}
   return C;
}

float** MakeMatrix(int size, float **restrict arr) {
    int i;
    arr = (float **)malloc( sizeof(float *) * size);
    arr[0] = (float *)malloc( sizeof(float) * size * size);

	//#pragma acc data copyout(arr[0:size])
    for (i=1; i<size; i++){
       arr[i] = (float *)(arr[i-1] + size);
    }
    return arr;
}
void showMatrix(int size, float **restrict arr) {
   int i, j;

   for (i=0; i<size; i++){
      for (j=0; j<size; j++){
         printf("arr[%d][%d]=%f \n",i,j,arr[i][j]);
      }
   }
}

void copyMatrix(float **restrict A, float **restrict B, int size){		
   for (int i=0; i<size; ++i){
      for (int j=0; j<size; ++j){
         A[i][j] = B[i][j];
      } 
   }
}
int main (int argc, char **argv) {
   int i, j, k;
   float **A, **B, **C, **temp1, **temp2;
     
   if (argc != 3) {
      fprintf(stderr,"Use: %s size nIter\n", argv[0]);
      return -1;
   }
   int size = atoi(argv[1]);
   int nIter = atoi(argv[2]);
    
   if (nIter <= 0) {
      fprintf(stderr,"%s: Invalid nIter (%d)\n", argv[0],nIter);
      return -1;
   }

    clock_t start_time, stop_time;  // timers --- change for MS

    A = (float**)MakeMatrix(size, A);
   fillMatrix(size, A);
	//#pragma acc data copyin(A[0:size])
   B = (float**)MakeMatrix(size, B);
   fillMatrix(size, B);
	//#pragma acc data copyin(A[0:size])
   C = (float**)MakeMatrix(size, C);

   //Storing array A and B to ensure they stay the same
   temp1 = A;
   temp2 = B;
   start_time = clock(); // Unix timer --- change for MS  	
	
   for (int i=0; i<nIter; i++) {
      C = MatrixMult(size, A, B, C);
      if (i%2==1) {
         copyMatrix(A, temp1, size); //multiply A by B and assign back to A on even iterations
      }
      else {
         copyMatrix(B, temp2, size); //multiply A by B and assign back to B on odd iterations
      }
   }

   stop_time = clock();  // timers --- change for MS
   float diff = ((float)(stop_time - start_time) / CLOCKS_PER_SEC);

   /*

   float diff = ( (stop_time.tv_sec-start_time.tv_sec)*1000000 + (stop_time.tv_usec - start_time.tv_usec) )/1000000.0;  // timers --- change for MS
   */

   /*
   showMatrix(size, A);
   showMatrix(size, B);
   showMatrix(size, C); 
   */
  
   printf("%s total runtime %8.5g\n", argv[0], diff);  

   /*
   free(A); free(B); free(C); 
   */
   return 0;
}

