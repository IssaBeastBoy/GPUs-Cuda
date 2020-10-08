/*************************************************
 * Laplace Serial C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  Copyright John Urbanic, PSC 2017
 *  Modified for Microsoft execution Karen Bradshaw, 2018
 ************************************************/
/*
	Average CPU time for 1000 -> 0.219000 seconds	

					for 10000 -> 25.92800 seconds	

					compilation: pgcc -acc -ta=host laplace_serial_RU.c 
					execution: laplace_serial_RU

	Average GPU time for 1000 (without copyin)	-> 1.14900 seconds 
							  (with copyin)		-> 0.279000 seconds 
							  speed up			-> Non
	
					for 10000 (without copyin)	-> 92.082000 seconds 
							  (with copyin)		-> 7.236000 seconds 
							  speed up			-> x 3.6

				complition: pgcc -acc -ta=tesla laplace_serial_RU.c
				execution: laplace_serial_RU

The CPU still runs faster then the GPU for the 1000 plate but the GPU runs faster when for 
10000 plate this is possibly due the GPU having more cores to use for parallelism as 
the data sizes increase resulting in the GPU out computing the CPU at high data size
	
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// size of plate
#define COLUMNS    1000
#define ROWS       1000

#ifndef MAX_ITER
#define MAX_ITER 100
#endif

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

double Temperature[ROWS+2][COLUMNS+2];      // temperature grid
double Temperature_last[ROWS+2][COLUMNS+2]; // temperature grid from last iteration

//   helper routines
void initialize();
void track_progress(int iter);


int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100;                                       // largest change in t
	clock_t start_time, stop_time;  		 // timers
	start_time = clock();
 
    max_iterations = MAX_ITER;

   //int gettimeofday(&start_time,NULL); // Unix timer

    initialize();                   // initialize Temp_last including boundary conditions

	// Data control pragma to move all the data require to GPU before executing 
	#pragma acc data copyin(Temperature, Temperature_last)
	{
		// do until error is minimal or until max steps
		while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) 
		{
			#pragma acc kernels
				// main calculation: average my four neighbors    
				for (i = 1; i <= ROWS; i++) {
					for (j = 1; j <= COLUMNS; j++) {
						Temperature[i][j] = 0.25 * (Temperature_last[i + 1][j] + Temperature_last[i - 1][j] +
							Temperature_last[i][j + 1] + Temperature_last[i][j - 1]);
					}
				}

				dt = 0.0; // reset largest temperature change
			#pragma	acc kernels 
			// copy grid to old grid for next iteration and find latest dt
				for (i = 1; i <= ROWS; i++) {
					for (j = 1; j <= COLUMNS; j++) {
						dt = fmax(fabs(Temperature[i][j] - Temperature_last[i][j]), dt);
						Temperature_last[i][j] = Temperature[i][j];
					}
				}

				// periodically print test values
				if ((iteration % 100) == 0) {
					track_progress(iteration);
				}

				iteration++;
		}
	}
	stop_time = clock();
	float amount_Oftime = ((float)(stop_time - start_time) / CLOCKS_PER_SEC);

	/*
    gettimeofday(&stop_time,NULL);
    float diff = ( (stop_time.tv_sec-start_time.tv_sec)*1000000 + (stop_time.tv_usec - start_time.tv_usec) )/1000000.0;
       */
    printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
    printf("Total time was %f seconds\n", amount_Oftime);

    exit(0);
}


// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(){

    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= ROWS+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= COLUMNS+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
    }
}


// print diagonal in bottom right corner where most action is
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = ROWS-5; i <= ROWS; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}
