#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"

// #define DEBUGGING_TEST 
// #define COMM_ALL_PROC

using namespace std;

extern double size;
#define BIN_SIZE (1.5 * 0.01) 

int BIN_SIDE;
int BINS;
double binSize; 

int get_bin(particle_t& p){
    int row = p.y / binSize;
    int col = p.x / binSize;
    return row * BIN_SIDE + col;
}  

#define get_row(idx) (idx / BIN_SIDE)
#define get_col(idx) (idx % BIN_SIDE)
#define get_Idx(r,c) (r * BIN_SIDE + c)

#define get_thread(_b) (max(min(_b / bins_per_thread, n_proc - 1),0))

bool out_range(int row, int col, int testI){
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
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
     //
    //  process command line parameters
    //
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
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    // int particle_per_proc = (n + n_proc - 1) / n_proc;
    // int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    // for( int i = 0; i < n_proc+1; i++ )
    //     partition_offsets[i] = min( i * particle_per_proc, n );
    
    // int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    // for( int i = 0; i < n_proc; i++ )
    //     partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
    
    //
    //  allocate storage for local partition
    //
    // int nlocal = partition_sizes[rank];
    // particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );

    // MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );

    MPI_Bcast( particles, n, PARTICLE, 0, MPI_COMM_WORLD );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    BIN_SIDE = ceil(size / BIN_SIZE);
    int rows_per_proc = max(BIN_SIDE / n_proc, 1);

    int row_bgn = rows_per_proc * rank;
    int row_end = min(row_bgn + rows_per_proc - 1, BIN_SIDE - 1);

    if(rank == n_proc - 1){
        row_end = BIN_SIDE - 1;
    }

    bool has_up_nb = rank > 0;
    bool has_dn_nb = rank < (n_proc - 1);

    if(row_end == BIN_SIDE - 1)
        has_dn_nb = false;

    if(row_bgn >= BIN_SIDE)
        has_up_nb = false;

    int row_nb_bgn = row_bgn - (has_up_nb ? 1: 0);
    int row_nb_end = row_end + (has_dn_nb ? 1: 0);

    int bin_bgn = row_bgn * BIN_SIDE;
    int bin_end = row_end * BIN_SIDE + BIN_SIDE - 1;

    int bin_nb_bgn = row_nb_bgn * BIN_SIDE;
    int bin_nb_end = row_nb_end * BIN_SIDE + BIN_SIDE - 1;

    vector< vector<particle_t> > bins[2];

    //init bins
    BIN_SIDE = ceil(size / BIN_SIZE);
    BINS =  (BIN_SIDE * BIN_SIDE);

    binSize = size / BIN_SIDE;

    int localBINS = max(bin_nb_end - bin_nb_bgn + 1,0);

    // if(row_nb_bgn >= BIN_SIDE)
    //     has_up_nb = has_dn_nb = false;

    bins[0].resize(localBINS);
    bins[1].resize(localBINS);

    int curBins = 0;
    int nextBins = 1;

    int neighbor[9] = {-BIN_SIDE - 1,-BIN_SIDE,-BIN_SIDE + 1,-1,0,1,BIN_SIDE - 1,BIN_SIDE,BIN_SIDE + 1};

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        #ifdef DEBUGGING_TEST
        printf("total bins = %d  per proc = %d\n", BIN_SIDE, rows_per_proc);
        fflush(stdout);
        #endif
    }

    for(int i = 0; i < n; i ++){
        int idx = get_bin(particles[i]);
        if(idx >= bin_nb_bgn && idx <= bin_nb_end)
            bins[curBins][idx - bin_nb_bgn].push_back(particles[i]);
    }

    free( particles );

    vector<particle_t> send_up[2];
    vector<particle_t> send_dn[2];
    vector<particle_t> recv_up(n);
    vector<particle_t> recv_dn(n);

    MPI_Request recv_up_req;
    MPI_Request recv_dn_req;
    MPI_Request send_up_req;
    MPI_Request send_dn_req;

    bool has_recv_up = false;
    bool has_recv_dn = false;

    #ifdef DEBUGGING_TEST
    for(int i = 0; i < n_proc; i ++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == rank){
            printf("rank: %d has %d localbins from %d to %d of row %d to %d up %d down %d\n", rank, localBINS, bin_nb_bgn, bin_nb_end, row_nb_bgn, row_nb_end, has_up_nb, has_dn_nb);
            // for(int j = 0; j < localBINS; j++){
            //     printf("bin %d has %d particles\n", j, bins[curBins][j].size());
            // }
            // printf("\n");
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    for( int step = 0; step < NSTEPS; step++ )
    {
        // printf("rank %d \n", );
        curBins = step & 1;
        nextBins = !curBins;

        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        // 
        //  collect all global data locally (not good idea to do)
        //
        // MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //

        
        // if( find_option( argc, argv, "-no" ) == -1 ){
        //     // if( fsave && (step%SAVEFREQ) == 0 ){
        //         MPI_Barrier(MPI_COMM_WORLD);
        //         vector<particle_t> particles;
        //         for(int row = row_bgn; row <= row_end; row++){
        //             for(int col = 0; col < BIN_SIDE; col++){
        //                 int idx = get_Idx(row,col);
        //                 int i = idx - row_nb_bgn;
        //                 for(vector<particle_t>::iterator it = bins[curBins][i].begin(); it != bins[curBins][i].end(); it++){
        //                     particle_t* newp = new particle_t(*it);
        //                     particles.push_back(*newp);
        //                 }
        //             }
        //         }

        //         if(rank != 0){
        //             int r = MPI_Send(particles.data(), particles.size(), PARTICLE, 0, 1, MPI_COMM_WORLD);
        //             assert(r == MPI_SUCCESS);
        //         }else{
        //             for(int i = 1; i < n_proc; i++){
        //                 MPI_Status status;
        //                 int cur_size = particles.size();
        //                 assert((n - cur_size) >= 0);
        //                 vector<particle_t> particles_i(n - cur_size);
        //                 int r = MPI_Recv(particles_i.data(), n - cur_size, PARTICLE, i, 1, MPI_COMM_WORLD, &status);
        //                 assert(r == MPI_SUCCESS);
        //                 int new_size;
        //                 MPI_Get_count(&status, PARTICLE, &new_size);
        //                 assert(new_size + cur_size <= n);
        //                 particles_i.resize(new_size);
        //                 particles.insert(particles.end(),particles_i.begin(), particles_i.end());
        //             }
        //             assert(particles.size() == n);

        //             particle_t *particles_a = (particle_t*) malloc( n * sizeof(particle_t) );
        //             for(int i = 0; i < n; i ++)
        //                 particles_a[i] = particles[i];

        //             save( fsave, n, particles_a );
        //         }
        //     // }
        // }


        
        //
        //  compute all forces
        //
        // for( int i = 0; i < nlocal; i++ )
        // {
        //     local[i].ax = local[i].ay = 0;
        //     for (int j = 0; j < n; j++ )
        //         apply_force( local[i], particles[j], &dmin, &davg, &navg );
        // }

        vector<particle_t>().swap(send_dn[curBins]);
        vector<particle_t>().swap(send_up[curBins]);

        #ifdef DEBUGGING_TEST
            int going_up = 0;
            int going_down = 0;
            int up_halo = 0;
            int down_halo = 0;
            int count = 0;
            int newCount = 0;
            int continue_up = 0;
            int continue_dn = 0;
        #endif

        // for(int i = bin_bgn; i <= bin_end; i++){
            // #ifdef DEBUGGING_TEST
            //     printf("rank: %d step: %d before receive bin %-4i has %-4i particles \n", rank, step, i, bins[curBins][i].size());
            // #endif   
        // }

        if(has_up_nb){
            if(step > 0 && !has_recv_up){
                MPI_Status status;
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d slow wait up for rank:%d\n", rank, step - 1, rank - 1);
                    //     fflush(stdout);
                    // #endif
                MPI_Wait(&recv_up_req, &status);
                int r_size;
                MPI_Get_count(&status, PARTICLE, &r_size);
                recv_up.resize(r_size);
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d slow receive %d particles from up %d\n", rank, step - 1, r_size, rank - 1);
                    //     fflush(stdout);
                    // #endif

                for(vector<particle_t>::iterator it = recv_up.begin(); it != recv_up.end(); it++){
                    int newI = get_bin(*it) - bin_nb_bgn;
                    bins[curBins][newI].push_back(*it);
                    // #ifdef DEBUGGING_TEST
                    //     printf("    rank: %d receive %-4f %-4f put bin %-4i\n", rank, it->x, it->y, newI);
                    // #endif                
                }
            }
            recv_up.resize(n);
            int r = MPI_Irecv(recv_up.data(), n, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, &recv_up_req);
            assert (r == MPI_SUCCESS);
            has_recv_up = false;
        }

 
         if(has_dn_nb){
                if(step > 0 && !has_recv_dn){
                    MPI_Status status;
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d slow wait dn for rank:%d\n", rank, step, rank + 1);
                    //     fflush(stdout);
                    // #endif
                    MPI_Wait(&recv_dn_req, &status);
                    int r_size;
                    MPI_Get_count(&status, PARTICLE, &r_size);
                    recv_dn.resize(r_size);
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d slow receive %d particles from down %d\n", rank, step - 1, r_size, rank + 1);
                    //     fflush(stdout);
                    // #endif
                    for(vector<particle_t>::iterator it = recv_dn.begin(); it != recv_dn.end(); it++){
                        int newI = get_bin(*it) - bin_nb_bgn;
                        bins[curBins][newI].push_back(*it);
                        // #ifdef DEBUGGING_TEST
                        //     printf("    rank: %d step %d receive particles %-4f %-4f from rank: %d\n", rank, step - 1, it->x, it->y, rank + 1);
                        //     fflush(stdout);
                        // #endif  
                    }
                }

            recv_dn.resize(n);
            int r = MPI_Irecv(recv_dn.data(), n, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, &recv_dn_req);
            assert (r == MPI_SUCCESS);
            has_recv_dn = false;
        }


        for(int row = row_bgn; row <= row_end; row++){

            for(int col = 0; col < BIN_SIDE; col++){

                int bin_idx = get_Idx(row, col);
                int i = bin_idx - bin_nb_bgn;

                // #ifdef DEBUGGING_TEST
                //     printf("rank: %d step: %d after receive bin %-4i has %-4i particles \n", rank, step, i, bins[curBins][i].size());
                // #endif   

                for(vector<particle_t>::iterator it = bins[curBins][i].begin(); it != bins[curBins][i].end(); it++){

                    #ifdef DEBUGGING_TEST
                        count++;
                    #endif
                    particle_t newp = *it;

                    newp.ax = 0;
                    newp.ay = 0;

                    for(int j = 0;j < 9; j ++){
                        int new_bin_idx = bin_idx + neighbor[j];
                        if(out_range(row, col, new_bin_idx)){
                            continue;
                        }
                        if(new_bin_idx >= 0 && new_bin_idx < BINS){
                            int newJ = new_bin_idx - bin_nb_bgn;
                            for(vector<particle_t>::iterator it_nb = bins[curBins][newJ].begin(); it_nb != bins[curBins][newJ].end(); it_nb++){
                                if(it_nb != it)
                                    apply_force(newp, *it_nb,&dmin,&davg,&navg);
                            }
                        }
                    }
                    move(newp); 
                    int idx = get_bin(newp);
                    int newRow = get_row(idx);

                    // printf("new row %d\n", newRow);

                    if(newRow <= row_bgn){
                        #ifdef DEBUGGING_TEST
                            if(newRow == row_bgn)
                                up_halo ++;
                            else
                                going_up ++;
                        #endif
                        send_up[curBins].push_back(newp);
                    }
                    if(newRow >= row_end){
                        #ifdef DEBUGGING_TEST
                            if(newRow == row_end)
                                down_halo ++;
                            else
                                going_down ++;
                        #endif
                        send_dn[curBins].push_back(newp);
                    }
                    if(newRow >= row_nb_bgn && newRow <= row_nb_end){
                        #ifdef DEBUGGING_TEST
                            newCount++;
                        #endif
                        bins[nextBins][idx - bin_nb_bgn].push_back(newp);
                    }
                }
            }

        }
     
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        if(has_up_nb){
            #ifdef COMM_ALL_PROC
            if(has_dn_nb){
                if(step > 0){
                    MPI_Status status;
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d wait dn for rank:%d\n", rank, step, rank + 1);
                    //     fflush(stdout);
                    // #endif
                    MPI_Wait(&recv_dn_req, &status);
                    int r_size;
                    MPI_Get_count(&status, PARTICLE, &r_size);
                    recv_dn.resize(r_size);
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d receive %d particles from down %d\n", rank, step, r_size, rank + 1);
                    //     fflush(stdout);
                    // #endif

                    has_recv_dn = true;

                    for(vector<particle_t>::iterator it = recv_dn.begin(); it != recv_dn.end(); it++){
                        int new_idx = get_bin(*it);
                        int newRow = get_row(new_idx);
                        assert(newRow <= row_nb_end);
                        if(newRow <= row_bgn){
                            send_up[curBins].push_back(*it);
                            #ifdef DEBUGGING_TEST
                                if(newRow != row_bgn)
                                    continue_up ++;
                            #endif   
                        }
                        
                        if(newRow >= row_nb_bgn){
                            #ifdef DEBUGGING_TEST
                                assert((new_idx - bin_nb_bgn) >= 0);
                                assert((new_idx - bin_nb_bgn) < localBINS);
                                newCount++;
                            #endif
                            bins[nextBins][new_idx - bin_nb_bgn].push_back(*it);
                        }
                        // #ifdef DEBUGGING_TEST
                        //     printf("    rank: %d step %d receive particles %-4f %-4f \n", rank, step, it->x, it->y);
                        //     fflush(stdout);
                        // #endif  
                    }
                }
            }
            #endif

            MPI_Status status;
            if(step > 0) MPI_Wait(&send_up_req, &status);
                    //  #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d send %d particles up to rank:%d\n", rank, step, send_up[curBins].size(), rank - 1);
                    //     fflush(stdout);
                    //     for(int i = 0; i < send_up[curBins].size(); i ++){
                    //         printf("   rank: %d step %d send %d up %-4f %-4f to rank: %d\n", rank, step, send_up[curBins].size(), send_up[curBins][i].x, send_up[curBins][i].y, rank - 1);
                    //         fflush(stdout);
                    //     }
                    // #endif

            int r = MPI_Isend(send_up[curBins].data(), send_up[curBins].size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD, &send_up_req);
            assert(r == MPI_SUCCESS);

        }

        if(has_dn_nb){
            #ifdef COMM_ALL_PROC
            if(has_up_nb){
                if(step > 0){
                    MPI_Status status;
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d wait up for rank:%d\n", rank, step, rank - 1);
                    //     fflush(stdout);
                    // #endif
                    MPI_Wait(&recv_up_req, &status);
                    int r_size;
                    MPI_Get_count(&status, PARTICLE, &r_size);
                    recv_up.resize(r_size);
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d receive %d particles from up %d\n", rank, step, r_size, rank - 1);
                    //     fflush(stdout);
                    // #endif
                    has_recv_up = true;

                    for(vector<particle_t>::iterator it = recv_up.begin(); it != recv_up.end(); it++){
                        int new_idx = get_bin(*it);
                        int newRow = get_row(new_idx);
                        assert(newRow >= row_nb_bgn);

                        if(newRow >= row_end){
                            send_dn[curBins].push_back(*it);
                            #ifdef DEBUGGING_TEST
                                if(newRow != row_end)
                                    continue_dn ++;
                            #endif   
                        }

                        if(newRow <= row_nb_end){
                            #ifdef DEBUGGING_TEST
                                assert((new_idx - bin_nb_bgn) >= 0);
                                assert((new_idx - bin_nb_bgn) < localBINS);
                                newCount++;
                            #endif
                            bins[nextBins][new_idx - bin_nb_bgn].push_back(*it);
                        }
                    }
                }
            }
            #endif

            MPI_Status status;
            if(step > 0) MPI_Wait(&send_dn_req, &status);
            int r = MPI_Isend(send_dn[curBins].data(), send_dn[curBins].size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD, &send_dn_req);
            assert(r == MPI_SUCCESS);
                    // #ifdef DEBUGGING_TEST
                    //     printf("rank: %d step %d send %d particles down to rank:%d\n", rank, step, send_dn[curBins].size(), rank + 1);
                    //     fflush(stdout);
                    // #endif
        }

        for(int row = row_nb_bgn; row <= row_nb_end; row ++)
            for(int col = 0; col < BIN_SIDE; col++)
                vector<particle_t>().swap(bins[curBins][get_Idx(row, col) - bin_nb_bgn]);


        #ifdef DEBUGGING_TEST
        MPI_Barrier(MPI_COMM_WORLD);
        int amount = 0;
        MPI_Reduce(&count, &amount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < n_proc; i++)
        {
            if (i == rank && localBINS != 0)
            {
                printf("%-4i %-4i going_up:%-4i up_halo:%-4i going_down:%-4i down_halo:%-4i old_count:%-4i new_count:%-4i continue_up: %-4i continue_dn: %-4i \n", rank, step, going_up, up_halo, going_down, down_halo, count, newCount, continue_up, continue_dn);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)
            assert(amount == n);
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
        //
        //  move particles
        //
        // for( int i = 0; i < nlocal; i++ )
        //     move( local[i] );
    }
    simulation_time = read_timer( ) - simulation_time;
  
    #ifdef DEBUGGING_TEST
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    if (rank == 0) {  
      // printf( "n = %d, simulation time = %g seconds", n, simulation_time);
      printf( "n = %d, processors = %d, simulation time = %g seconds", n, n_proc, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    // free( partition_offsets );
    // free( partition_sizes );
    // free( local );

    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
