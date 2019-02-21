#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "common.h"

using namespace std;
//
//  benchmarking program
//

extern double size;

#define BIN_SIZE (1.5 * 0.01) 

int BIN_SIDE;
int BINS;
// #define MAX_PER_BIN(n) (n * 4 / BINS)
double binSize; 

// //bin and p_list
// typedef struct 
// {
//     particle_t* p;
//     p_node* next;
// }p_node;

// typedef struct
// {
//     p_node* head;  
//     p_node* end;
//     bin(){
//         head = NULL;
//         end = head;
//     }
//     void push_back(particle_t* p){
//         if(head == NULL){
//             head = p;
//             end = head;
//         }else{
//             end -> next = p;
//             end = end -> next;
//         }
//     }
// }bin;

int get_bin(particle_t& p){
    int row = p.y / binSize;
    int col = p.x / binSize;
    return row * BIN_SIDE + col;
}

bool out_range(int curI, int testI){
    int row = curI / BIN_SIDE;
    int col = curI % BIN_SIDE;
    int trow = testI / BIN_SIDE;
    int tcol = testI % BIN_SIDE;
    if(abs(tcol - col) > 1 || abs(trow - row) > 1)
        return true;
    else
        return false;
}

int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }

    printf("NSTEPS = %d\n",NSTEPS);
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    

    //before or after ??
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    //add bins
    // list<particle_t*> bins[BINS];

    BIN_SIDE = ceil(size / BIN_SIZE);
    BINS =  (BIN_SIDE * BIN_SIDE);
    
    vector< vector<particle_t*> > bins[2];

    binSize = size / BIN_SIDE;//get_size();
    
    // printf("init bins. \n");
    bins[0].assign(BINS, vector<particle_t*>());
    bins[1].assign(BINS, vector<particle_t*>());

    int curBins = 0;
    int nextBins = 1;

    for(int i = 0; i < n; i ++){
        int idx = get_bin(particles[i]);
        bins[curBins][idx].push_back(&particles[i]);
        // printf("in %d bin.\n", idx);
    }

    // for(int i = 0; i < BINS; i ++)
    //     bins[1].clear();
    // printf("end init bins. \n");


    int neighbor[9] = {-BIN_SIDE - 1,-BIN_SIDE,-BIN_SIDE + 1,-1,0,1,BIN_SIDE - 1,BIN_SIDE,BIN_SIDE + 1};


    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;
    //     //
    //     //  compute forces
    //     //
    //     for( int i = 0; i < n; i++ )
    //     {
    //         particles[i].ax = particles[i].ay = 0;
    //         for (int j = 0; j < n; j++ )
                // apply_force( particles[i], particles[j],&dmin,&davg,&navg);
    //     }

        // printf("generate bins. \n");
        // compute bins
        for(int i = 0; i < BINS; i ++){
            // cout<<"bins["<<i<<"].size() = "<< bins[curBins][i].size()<<endl;
            for(vector<particle_t*>::iterator it = bins[curBins][i].begin(); it != bins[curBins][i].end(); it++){
                (*it) -> ax = 0;
                (*it) -> ay = 0;
                for(int j = 0;j < 9; j ++){
                    int newJ = i + neighbor[j];
                    if(out_range(i, newJ)){
                        continue;
                    }
                    if(newJ >= 0 && newJ < BINS){
                        for(vector<particle_t*>::iterator it_nb = bins[curBins][newJ].begin(); it_nb != bins[curBins][newJ].end(); it_nb++){
                            apply_force( **it, **it_nb,&dmin,&davg,&navg);
                        }
                    }
                }
            }
        }

        // for (int i = 0; i < BIN_SIDE; i++){

            
        // }
        
        // //
        // //  move particles
        // //
        // for( int i = 0; i < n; i++ ){
        //     move( particles[i] );    
        // }

        // printf("move and generate newbin. \n");
        //swich buffer
        for(int i = 0; i < BINS; i ++){
            for(vector<particle_t*>::iterator it = bins[curBins][i].begin(); it != bins[curBins][i].end(); it++){
                move(**it);   
                int newbin = get_bin(**it);
                bins[nextBins][newbin].push_back(*it);
            } 
            bins[curBins][i].clear(); 
        }

        curBins = nextBins;
        nextBins = curBins == 1? 0: 1;
        // cout<<curBins<<"  ____________________   "<<nextBins<<endl;

        // printf("finish. \n");

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
        
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will havea particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}