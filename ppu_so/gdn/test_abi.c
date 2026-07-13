#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int ppu_gdn_recurrent(const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,int,int,int,int,int,float,void*);
static float* rd(const char*p,long*n){FILE*f=fopen(p,"rb");if(!f){printf("no %s\n",p);exit(1);}fseek(f,0,SEEK_END);*n=ftell(f)/4;fseek(f,0,SEEK_SET);float*h=malloc(*n*4);fread(h,4,*n,f);fclose(f);return h;}
static float* up(float*h,long n){void*d;cudaMalloc(&d,n*4);cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice);return d;}
int main(){
  char p[512]; long n;
  #define L(nm) (sprintf(p,"%s/%s.bin","GOLDDIR",#nm), rd(p,&n))
  #define GOLDDIR "golden"
  long nq,nk,nv,ng,nb,nh0,no,nht;
  sprintf(p,"golden/q.bin");   float*q=rd(p,&nq);
  sprintf(p,"golden/k.bin");   float*k=rd(p,&nk);
  sprintf(p,"golden/v.bin");   float*v=rd(p,&nv);
  sprintf(p,"golden/g.bin");   float*g=rd(p,&ng);
  sprintf(p,"golden/beta.bin");float*b=rd(p,&nb);
  sprintf(p,"golden/h0.bin");  float*h0=rd(p,&nh0);
  sprintf(p,"golden/o.bin");   float*oref=rd(p,&no);
  sprintf(p,"golden/ht.bin");  float*htref=rd(p,&nht);
  int B=1,T=8,H=2,HV=4,S=128; float scale=0.08838834764831843f;
  float*dq=up(q,nq),*dk=up(k,nk),*dv=up(v,nv),*dg=up(g,ng),*db=up(b,nb),*dh0=up(h0,nh0);
  float*d_o;cudaMalloc(&d_o,no*4);float*d_ht;cudaMalloc(&d_ht,nht*4);
  cudaMemset(d_o,0,no*4);cudaMemset(d_ht,0,nht*4);
  int rc=ppu_gdn_recurrent(dq,dk,dv,dg,db,dh0,d_o,d_ht,B,T,H,HV,S,scale,(void*)0);
  cudaDeviceSynchronize();
  float*o=malloc(no*4),*ht=malloc(nht*4);
  cudaMemcpy(o,d_o,no*4,cudaMemcpyDeviceToHost);cudaMemcpy(ht,d_ht,nht*4,cudaMemcpyDeviceToHost);
  double so=0,no2=0,sh=0,nh2=0;
  for(long i=0;i<no;i++){double d=o[i]-oref[i];so+=d*d;no2+=oref[i]*oref[i];}
  for(long i=0;i<nht;i++){double d=ht[i]-htref[i];sh+=d*d;nh2+=htref[i]*htref[i];}
  double ro=sqrt(so/(no2+1e-12)),rh=sqrt(sh/(nh2+1e-12));
  printf("rc=%d  o rel_rms=%.3e  ht rel_rms=%.3e -> %s\n",rc,ro,rh,(rc==0&&ro<1e-4&&rh<1e-4)?"PASS":"FAIL");
  return (rc==0&&ro<1e-4&&rh<1e-4)?0:1;
}
