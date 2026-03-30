// Unit test: tq3_prefill_kernel correctness
// Verifies: WHT-once-per-block × N tokens matches CPU reference

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_TQ3_0 32
typedef struct { __half d; uint8_t qs[12]; } block_tq3_0;

// CPU reference
static float cpu_sign(int i) { return ((((unsigned)i*0x9E3779B9u)>>31)&1)?-1.0f:1.0f; }
static const float C[8]={-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};
static const float B[7]={-1.7479f,-1.0500f,-0.5005f,0.0f,0.5005f,1.0500f,1.7479f};

static void cpu_quantize_tq3(const float *x, block_tq3_0 *blk) {
    float sq=0; for(int i=0;i<32;i++) sq+=x[i]*x[i];
    float rms=sqrtf(sq/32.0f); if(rms<1e-10f) rms=1.0f;
    blk->d=__float2half(rms);
    float buf[32]; for(int i=0;i<32;i++) buf[i]=x[i]/rms*cpu_sign(i);
    for(int s=1;s<32;s<<=1) for(int i=0;i<32;i+=s*2) for(int j=i;j<i+s;j++){float a=buf[j],b=buf[j+s];buf[j]=a+b;buf[j+s]=a-b;}
    for(int i=0;i<32;i++) buf[i]/=sqrtf(32.0f);
    uint8_t idx[32]; for(int i=0;i<32;i++){idx[i]=0;for(int b=0;b<7;b++) if(buf[i]>B[b]) idx[i]=b+1;}
    for(int g=0;g<4;g++){uint8_t*q=blk->qs+g*3,*d=idx+g*8;
        q[0]=d[0]|(d[1]<<3)|(d[2]<<6);q[1]=(d[2]>>2)|(d[3]<<1)|(d[4]<<4)|(d[5]<<7);q[2]=(d[5]>>1)|(d[6]<<2)|(d[7]<<5);}
}

static void cpu_dequant_tq3(const block_tq3_0 *blk, float *out) {
    float rms=__half2float(blk->d);
    float v[32];
    for(int g=0;g<4;g++){const uint8_t*q=blk->qs+g*3;int b=g*8;
        v[b+0]=C[q[0]&7];v[b+1]=C[(q[0]>>3)&7];v[b+2]=C[((q[0]>>6)|(q[1]<<2))&7];
        v[b+3]=C[(q[1]>>1)&7];v[b+4]=C[(q[1]>>4)&7];v[b+5]=C[((q[1]>>7)|(q[2]<<1))&7];
        v[b+6]=C[(q[2]>>2)&7];v[b+7]=C[(q[2]>>5)&7];}
    for(int s=1;s<32;s<<=1) for(int i=0;i<32;i+=s*2) for(int j=i;j<i+s;j++){float a=v[j],b=v[j+s];v[j]=a+b;v[j+s]=a-b;}
    for(int i=0;i<32;i++) out[i]=v[i]/sqrtf(32.0f)*cpu_sign(i)*rms;
}

// CPU reference matmul: weights[ne01][ne00] × act[ne11][ne00] → dst[ne01][ne11]
static void cpu_matmul(const block_tq3_0 *weights, const float *act,
                       float *dst, int ne00, int ne01, int ne11) {
    int nb = ne00 / 32;
    for (int row = 0; row < ne01; row++) {
        for (int tok = 0; tok < ne11; tok++) {
            float sum = 0;
            for (int blk = 0; blk < nb; blk++) {
                float dq[32];
                cpu_dequant_tq3(&weights[row*nb+blk], dq);
                for (int j = 0; j < 32; j++)
                    sum += dq[j] * act[tok*ne00 + blk*32+j];
            }
            dst[row*ne11+tok] = sum;
        }
    }
}

// GPU kernel (copy from tq3-prefill.cuh)
#define TILE_N 8
__constant__ float GPU_C[8]={-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};

__device__ float gpu_sign(int i){return((((unsigned)i*0x9E3779B9u)>>31)&1)?-1.0f:1.0f;}
__device__ float gpu_centroid(uint8_t idx){return GPU_C[idx&7];}
__device__ uint8_t gpu_unpack(uint32_t p,int r){return(p>>(3*r))&7;}

__global__ void tq3_prefill(const block_tq3_0*W,const float*A,float*D,int ne00,int ne01,int ne11){
    int row=blockIdx.x,tt=blockIdx.y,lane=threadIdx.x;
    if(row>=ne01) return;
    int nb=ne00/32,t0=tt*TILE_N,t1=min(t0+TILE_N,ne11);
    float acc[TILE_N]={};
    for(int blk=0;blk<nb;blk++){
        const block_tq3_0*bq=W+row*nb+blk;
        float rms=__half2float(bq->d);
        int g=lane/8,r=lane%8,leader=g*8;
        uint32_t packed=0;
        if(r==0){const uint8_t*qp=bq->qs+g*3;packed=(uint32_t)qp[0]|((uint32_t)qp[1]<<8)|((uint32_t)qp[2]<<16);}
        packed=__shfl_sync(0xFFFFFFFF,packed,leader);
        float val=gpu_centroid(gpu_unpack(packed,r));
        #pragma unroll
        for(int step=1;step<32;step<<=1){float o=__shfl_xor_sync(0xFFFFFFFF,val,step);val=(lane&step)?(o-val):(o+val);}
        float wj=val*gpu_sign(lane)*(rms/sqrtf(32.0f));
        #pragma unroll
        for(int t=0;t<TILE_N;t++){int tok=t0+t;if(tok<ne11) acc[t]+=wj*A[tok*ne00+blk*32+lane];}
    }
    #pragma unroll
    for(int t=0;t<TILE_N;t++){
        float s=acc[t];
        #pragma unroll
        for(int m=16;m>0;m>>=1) s+=__shfl_xor_sync(0xFFFFFFFF,s,m);
        if(lane==0){int tok=t0+t;if(tok<ne11) D[row*ne11+tok]=s;}
    }
}

int main(){
    printf("=== TQ3_0 native prefill kernel test ===\n\n");
    const int ne00=128, ne01=16, ne11=8; // small: 16 weight rows, 8 tokens
    const int nb=ne00/32;

    // Build weight blocks and activations
    block_tq3_0 *h_W=(block_tq3_0*)malloc(ne01*nb*sizeof(block_tq3_0));
    float *h_A=(float*)malloc(ne11*ne00*sizeof(float));
    float *h_cpu=(float*)malloc(ne01*ne11*sizeof(float));
    float *h_gpu=(float*)malloc(ne01*ne11*sizeof(float));

    for(int i=0;i<ne01*nb;i++){
        float tmp[32]; for(int j=0;j<32;j++) tmp[j]=sinf(i*100+j*0.3f+1.0f)*0.5f;
        cpu_quantize_tq3(tmp,&h_W[i]);
    }
    for(int i=0;i<ne11*ne00;i++) h_A[i]=cosf(i*0.07f)*0.3f;

    // CPU reference
    cpu_matmul(h_W,h_A,h_cpu,ne00,ne01,ne11);

    // GPU
    block_tq3_0*d_W; float*d_A,*d_D;
    cudaMalloc(&d_W,ne01*nb*sizeof(block_tq3_0));
    cudaMalloc(&d_A,ne11*ne00*4);
    cudaMalloc(&d_D,ne01*ne11*4);
    cudaMemcpy(d_W,h_W,ne01*nb*sizeof(block_tq3_0),cudaMemcpyHostToDevice);
    cudaMemcpy(d_A,h_A,ne11*ne00*4,cudaMemcpyHostToDevice);

    dim3 grid(ne01,(ne11+TILE_N-1)/TILE_N);
    tq3_prefill<<<grid,32>>>(d_W,d_A,d_D,ne00,ne01,ne11);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu,d_D,ne01*ne11*4,cudaMemcpyDeviceToHost);

    // Compare
    float maxerr=0; int fail=0;
    for(int i=0;i<ne01*ne11;i++){
        float e=fabsf(h_cpu[i]-h_gpu[i])/(fabsf(h_cpu[i])+1e-6f);
        if(e>maxerr) maxerr=e;
        if(e>0.01f) fail++;
    }
    printf("Max relative error: %.4f%%\n",maxerr*100);
    printf("Elements with >1%% error: %d / %d\n",fail,ne01*ne11);
    printf("Result: %s\n\n",fail==0?"PASS":"FAIL");

    free(h_W);free(h_A);free(h_cpu);free(h_gpu);
    cudaFree(d_W);cudaFree(d_A);cudaFree(d_D);
    return fail>0?1:0;
}
