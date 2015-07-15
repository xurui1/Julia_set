/*
Code adapted from book "CUDA by Example: An Introduction to General-Purpose GPU Programming" 

This code computes a visualization of the Julia set.  Two-dimensional "bitmap" data which can be plotted is computed by the function kernel.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

This code is CPU only but will compile with:

nvcc julia_cpu.cu

 
*/


#include <stdio.h>
#include <cuda.h>

#define DIM 1000

__device__ int julia( int x, int y ) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    float cr=-0.8f;
    float ci=0.156f;

    float ar=jx;
    float ai=jy;
    float artmp;

    int i = 0;
    for (i=0; i<200; i++) {

        artmp = ar;
        ar =(ar*ar-ai*ai) +cr;
        ai = 2.0f*artmp*ai + ci;

        if ( (ar*ar+ai*ai) > 1000)
            return 0;
    }

    return 1;
}

__global__ void  kernel( int *arr_d, int n ){
  

  int x,y;
  x=blockIdx.x * blockDim.x + threadIdx.x;
  y=blockIdx.y * blockDim.y + threadIdx.y;
 
  int offset = x + y * DIM;

  int juliaValue = julia( x, y );
  arr_d[offset] = juliaValue;
        
    
}

int main( void ) {
    int *arr_h;
    int *arr_d;
    int *arr_shadow;
    
    int n = DIM*DIM;
    size_t memsize; 
    memsize = n * sizeof(int); 



    arr_h = (int *)malloc(memsize);
    cudaMalloc((void **) &arr_d, memsize);
    arr_shadow = (int *)malloc(memsize);
    
    cudaMemcpy(arr_h,arr_d,memsize, cudaMemcpyHostToDevice);   
    dim3 gridDef1(DIM,DIM,1);
    dim3 blockDef1(1,1,1); 
    
    FILE *out;
 
    //execute kernel
    kernel<<<gridDef1,blockDef1>>>(arr_d, n); 

    //Retrieve results
    cudaMemcpy(arr_shadow, arr_d, memsize, cudaMemcpyDeviceToHost); 

    //Ensure synchronization
    cudaDeviceSynchronize(); 

    out = fopen( "julia.dat", "w" );
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(arr_shadow[offset]==1)
                fprintf(out,"%d %d \n",x,y);  
        } 
    } 
    fclose(out);

}

