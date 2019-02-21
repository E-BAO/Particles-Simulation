#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <list>
#include "common.h"
#include "omp.h"

using namespace std;

extern double size;

#define BIN_SIZE (1.5 * 0.01) 

static int BIN_SIDE;
static int BINS;
static double binSize; 
static vector< vector<particle_t*> > bins[2];
static vector<omp_lock_t> locks;

static int get_bin(particle_t& p){
    int row = p.y / binSize;
    int col = p.x / binSize;
    return row * BIN_SIDE + col;
}


#define get_thread(_b) (max(min(_b / bins_per_thread, numthreads - 1),0))

// #define CLAMP(_v, _l, _h) ( (_v) < (_l) ? (_l) : (((_v) > (_h)) ? (_h) : (_v)) )
// #define get_thread_for_bin(_b) CLAMP((_b) / bins_per_thread, 0, numthreads-1)

// int get_thread(int bin_id){
//     return 
// }

static bool out_range(int row, int col, int testI){
    int trow = testI / BIN_SIDE;
    int tcol = testI % BIN_SIDE;
    if(abs(tcol - col) > 1 || abs(trow - row) > 1)
        return true;
    else
        return false;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
    
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    //init bins
    BIN_SIDE = ceil(size / BIN_SIZE);
    BINS =  (BIN_SIDE * BIN_SIDE);
    // vector< list<particle_t*> > bins;
    // binSize = size / BIN_SIDE;
    // bins.resize(BINS);

    binSize = size / BIN_SIDE;//get_size();

    // printf("init bins = %d. \n", BINS);

    bins[0].resize(BINS);
    bins[1].resize(BINS);

    int curBins = 0;
    int nextBins = 1;

    int neighbor[9] = {-BIN_SIDE - 1,-BIN_SIDE,-BIN_SIDE + 1,-1,0,1,BIN_SIDE - 1,BIN_SIDE,BIN_SIDE + 1};


    // omp_lock_t* locks = (omp_lock_t*) malloc(BINS * sizeof(omp_lock_t) );
    // for (int i = 0; i < BINS; i++) {
    //     omp_init_lock(locks+i);
    // }
    
    // vector<omp_lock_t> t_locks;
    vector< vector<particle_t*> > thread_block;

    #pragma omp parallel
    {
        numthreads = omp_get_num_threads();
    }

    locks.resize(numthreads);
    for(int i = 0; i < locks.size(); i ++){
        omp_init_lock(&locks[i]);
    }
    // t_locks.resize()
    thread_block.resize(numthreads);

    int bins_per_thread = BINS / numthreads;

    for(int i = 0; i < n; i ++){
        int idx = get_bin(particles[i]);
        int t_id = get_thread(idx);
        thread_block[t_id].push_back(&particles[i]);
    }

    #pragma omp parallel private(dmin)
    {
        int thread_num = omp_get_thread_num();
        // cout<<"thread_num="<<thread_num<<endl;


        for(vector<particle_t*> ::iterator it = thread_block[thread_num].begin(); it != thread_block[thread_num].end(); it ++){
            int idx = get_bin(**it);
            // particle_t* newp =  new particle_t(**it);
            bins[curBins][idx].push_back(*it);
        }
        thread_block[thread_num].resize(0);
        #pragma omp barrier
        
        for( int step = 0; step < NSTEPS; step++ )
        {
            curBins = step & 1;
            nextBins = !curBins;
            // printf(" t%d switch  %d\n",thread_num, curBins);
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            // printf("%d\n", thread_num);
            // int amount = 0;

            #pragma omp for schedule(static, bins_per_thread)
            for(int i = 0; i < BINS; i ++){
                bins[nextBins][i].resize(0);
                // amount += bins[curBins][i].size();
                // printf("               t%d old%d\n", thread_num, bins[curBins][i].size());
            }
            // printf("thread%d  switch%d step%d  particles%d  \n", thread_num, curBins, step, amount);
            // #pragma omp barrier

            // int stay = 0;
            // int jump = 0;
            #pragma omp for schedule(static, bins_per_thread) reduction (+:navg) reduction(+:davg)
            for(int i = 0; i < BINS; i ++){
                int row = i / BIN_SIDE;
                int col = i % BIN_SIDE;
                // printf("gogo\n");
                for(vector<particle_t*>::iterator it = bins[curBins][i].begin(); it != bins[curBins][i].end(); it++){
                    // printf("yeah\n");
                    (*it) -> ax = 0;
                    (*it) -> ay = 0;
                    for(int j = 0;j < 9; j ++){
                        int newJ = i + neighbor[j];
                        if(out_range(row, col, newJ)){
                            continue;
                        }
                        if(newJ >= 0 && newJ < BINS){
                            for(vector<particle_t*>::iterator it_nb = bins[curBins][newJ].begin(); it_nb != bins[curBins][newJ].end(); it_nb++){
                                apply_force( **it, **it_nb,&dmin,&davg,&navg);
                            }
                        }
                    }
                    particle_t* newp = new particle_t(**it);
                    move(*newp); 
                    int idx = get_bin(*newp);
                    if(idx == i){
                        // particle_t* newpp = new particle_t(*newp);
                        bins[nextBins][i].push_back(newp);
                        // stay ++;
                        // printf("  stay  t%d  s%d  b%d  n%d\n", thread_num, step, i, bins[curBins][i].size());
                    }else{
                        int t_id = get_thread(idx);
                        // list<particle_t*>::iterator it_l = it;
                        // it --;
                        omp_set_lock(&locks[t_id]);
                        thread_block[t_id].push_back(newp);
                        omp_unset_lock(&locks[t_id]);
                        // jump ++;
                        // printf("  jump  t%d  s%d  b%d  n%d\n", thread_num, step, i, bins[curBins][i].size());
                    }
                }
            }
            #pragma omp barrier

            // #pragma omp for schedule(static, bins_per_thread)
            // for(int i = 0; i < BINS; i ++){
            //     printf("               t%d new%d\n", thread_num, bins[nextBins][i].size());
            // }
            // #pragma omp barrier

            for(vector<particle_t*> ::iterator it = thread_block[thread_num].begin(); it != thread_block[thread_num].end(); it ++){
                int idx = get_bin(**it);
                bins[nextBins][idx].push_back(*it);
            }
            thread_block[thread_num].resize(0);
            // if(thread_num == 0){
            //     nextBins = curBins;
            //     curBins = nextBins == 1? 0: 1;
            // }
            // #pragma omp barrier

            // #pragma omp for schedule(static, bins_per_thread)
            // for(int i = 0; i < BINS; i ++){
            //     printf("               t%d newafter%d\n", thread_num, bins[nextBins][i].size());
            // }
            // #pragma omp barrier


            // #pragma omp for schedule(static, bins_per_thread)
            // for(int i = 0; i < BINS; i ++){
            //     printf("               t%d old%d\n", thread_num, bins[curBins][i].size());
            // }
            // #pragma omp barrier


            // printf("       t%d stay%d  jump%d\n", thread_num, stay, jump);

            // #pragma omp barrier

            //
            //move particles and clear buffermove particles
            //
            // for(int i = 0; i < BINS; i ++){
            //     for(vector<particle_t*>::iterator it = bins[i].begin(); it != bins[i].end(); it++){
            //         particle_t* newp = new particle_t(**it);
            //         move(newp); 
            //         int idx = get_bin(newp);
            //         if(idx == i){
            //             bins[nextBins][i].push_back(newp);
            //             continue;
            //         }else{
            //             int t_id = get_thread(idx);
            //             // list<particle_t*>::iterator it_l = it;
            //             // it --;
            //             omp_set_lock(&locks[t_id]);
            //             // thread_block[t_id].splice(thread_block[t_id].begin(), bins[i], it_l);
            //             thread_block[t_id].push_back(newp);
            //             omp_unset_lock(&locks[t_id]);
            //         }
            //     } 
            //     // bins[i].clear(); 
            // }
            // #pragma omp barrier


            // for(list<particle_t*> ::iterator it = thread_block[thread_num].begin(); it != thread_block[thread_num].end(); it ++){
            //     int idx = get_bin(**it);
            //     // omp_set_lock(&locks[idx]);
            //     bins[idx].push_back(*it);
            //     // omp_unset_lock(&locks[idx]);
            // }
            // thread_block[thread_num].clear();
            // #pragma omp barrier
            


            if( find_option( argc, argv, "-no" ) == -1 ) 
            {
              //
              //  compute statistical data
              //
              #pragma omp master
              if (navg) { 
                absavg += davg/navg;
                nabsavg++;
              }

              #pragma omp critical
              if (dmin < absmin) absmin = dmin; 
            
              //
              //  save if necessary
              //
              #pragma omp master
              if( fsave && (step%SAVEFREQ) == 0 )
                  save( fsave, n, particles );
            }
            #pragma omp barrier
        }
    }

    for(int i = 0; i < locks.size(); i ++){
        omp_destroy_lock(&locks[i]);
    }
    
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
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
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

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
