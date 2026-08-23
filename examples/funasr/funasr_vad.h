// funasr_vad.h — single-header FSMN-VAD for the funasr ggml runtime.
// Exposes funasr_vad_segments(): 16k mono wav -> speech segments [start_ms,end_ms].
// Front end (80-mel fbank + LFR m5n1 + CMVN) and FSMN encoder validated bit-exact vs
// PyTorch fsmn-vad; the host state machine reproduces E2EVadModel (DEFAULT_SILENCE_SCHEDULE,
// chunk-stepped) to within 1 frame (10ms) of fsmn-vad.generate on the 184-clip set.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846   // not guaranteed by <cmath> on MSVC
#endif

namespace funasr_vad_impl {
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
    float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}}
static std::vector<std::vector<float>> fbank80(std::vector<float> wav){
  for(auto&v:wav)v*=32768.0f; std::vector<float>win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NB=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>>fb(NMEL,std::vector<float>(NB,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;for(int k=0;k<NB;k++){float mf=melf(bw*k);if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; if(T<1)T=0; std::vector<std::vector<float>>feat(T,std::vector<float>(NMEL));
  std::vector<float>re(NFFT),im(NFFT),fr(WINLEN);const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NB;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);feat[t][m]=logf(e>fl?e:fl);}}
  return feat;
}
static std::vector<float> lfr(const std::vector<std::vector<float>>&feat,int m,int n,int&T_out){
  int T=feat.size(); if(T<1){T_out=0;return {};}     // empty (audio shorter than one frame)
  int D=NMEL,pad=(m-1)/2; int Tl=(T+n-1)/n;
  std::vector<std::vector<float>> pf; pf.reserve(T+pad+m);
  for(int i=0;i<pad;i++)pf.push_back(feat[0]);
  for(int t=0;t<T;t++)pf.push_back(feat[t]);
  while((int)pf.size()<(Tl-1)*n+m)pf.push_back(feat[T-1]);
  std::vector<float> out((size_t)Tl*m*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<m;j++)memcpy(&out[((size_t)i*m+j)*D],pf[i*n+j].data(),D*sizeof(float));
  T_out=Tl; return out;
}
struct vad{ggml_context*ctx=nullptr;std::map<std::string,ggml_tensor*>t;
  ggml_tensor*g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"vad: missing %s\n",n.c_str());return nullptr;}return it->second;}};
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
} // namespace funasr_vad_impl

// Run FSMN-VAD on a 16k mono float waveform; fills segs with [start_ms,end_ms] speech spans.
// max_seg_ms caps a single segment (e.g. 30000); pass <=0 to use 60000 (model default).
inline bool funasr_vad_segments(const std::string& gguf_path, const std::vector<float>& wav,
                                int max_seg_ms, std::vector<std::pair<int,int>>& segs, int nthreads=8){
  using namespace funasr_vad_impl;
  segs.clear();
  vad m; gguf_init_params ip={false,&m.ctx}; gguf_context*gg=gguf_init_from_file(gguf_path.c_str(),ip);
  if(!gg){fprintf(stderr,"vad: cannot load %s\n",gguf_path.c_str());return false;}
  auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  int idim=rd("vad.input_dim",400),pd=rd("vad.proj_dim",128),nl=rd("vad.fsmn_layers",4),lorder=rd("vad.lorder",20),
      od=rd("vad.output_dim",248),lm=rd("vad.lfr_m",5),ln=rd("vad.lfr_n",1);
  for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx,nm);}
  gguf_free(gg);

  // fail fast (not segfault) if the GGUF is missing tensors the graph dereferences
  auto need=[&](const std::string&n){ return m.g(n)!=nullptr; };
  bool ok_t = need("cmvn.shift")&&need("cmvn.scale")&&need("encoder.in_linear1.linear.weight")
            &&need("encoder.in_linear2.linear.weight")&&need("encoder.out_linear1.linear.weight")
            &&need("encoder.out_linear2.linear.weight");
  for(int i=0;i<nl&&ok_t;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
    ok_t=need(p+"linear.linear.weight")&&need(p+"fsmn_block.conv_left.weight")&&need(p+"affine.linear.weight");}
  if(!ok_t){fprintf(stderr,"vad: gguf missing required tensors\n"); if(m.ctx)ggml_free(m.ctx); return false;}

  auto feat=fbank80(wav); int T=0; auto feats=lfr(feat,lm,ln,T);   // [T,400]
  if(T<1){if(m.ctx)ggml_free(m.ctx);return true;}                  // too short -> no speech
  float*shift=(float*)m.g("cmvn.shift")->data,*scale=(float*)m.g("cmvn.scale")->data;
  for(int t=0;t<T;t++)for(int d=0;d<idim;d++)feats[(size_t)t*idim+d]=(feats[(size_t)t*idim+d]+shift[d])*scale[d];

  ggml_backend_t be=ggml_backend_cpu_init();
  // no_alloc=true -> ctx holds only tensor/graph metadata (the real compute buffer is
  // allocated by gallocr below), so a few MB is plenty regardless of clip length.
  ggml_init_params cp={(size_t)16*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
  ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,idim,T); ggml_set_input(x);
  ggml_tensor*h=lin(c,m.g("encoder.in_linear1.linear.weight"),m.g("encoder.in_linear1.linear.bias"),x);
  h=lin(c,m.g("encoder.in_linear2.linear.weight"),m.g("encoder.in_linear2.linear.bias"),h); h=ggml_relu(c,h);
  for(int i=0;i<nl;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
    ggml_tensor*z=ggml_mul_mat(c,m.g(p+"linear.linear.weight"),h);
    ggml_tensor*fk=m.g(p+"fsmn_block.conv_left.weight"); ggml_tensor*zp=ggml_pad_ext(c,z,0,0,lorder-1,0,0,0,0,0); ggml_tensor*acc=z;
    // sl is a full-row slice of the contiguous padded tensor -> already contiguous, no ggml_cont needed
    for(int j=0;j<lorder;j++){auto sl=ggml_view_2d(c,zp,pd,T,zp->nb[1],(size_t)j*zp->nb[1]);auto wj=ggml_view_1d(c,fk,pd,(size_t)j*fk->nb[1]);acc=ggml_add(c,acc,ggml_mul(c,sl,wj));}
    ggml_tensor*a=lin(c,m.g(p+"affine.linear.weight"),m.g(p+"affine.linear.bias"),acc); h=ggml_relu(c,a);}
  h=lin(c,m.g("encoder.out_linear1.linear.weight"),m.g("encoder.out_linear1.linear.bias"),h);
  h=lin(c,m.g("encoder.out_linear2.linear.weight"),m.g("encoder.out_linear2.linear.bias"),h);
  h=ggml_soft_max(c,h); ggml_set_output(h);
  ggml_cgraph*gf=ggml_new_graph(c); ggml_build_forward_expand(gf,h);
  ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
  ggml_backend_tensor_set(x,feats.data(),0,ggml_nbytes(x)); ggml_backend_cpu_set_n_threads(be,nthreads);
  bool ok=ggml_backend_graph_compute(be,gf)==GGML_STATUS_SUCCESS;
  std::vector<float> sc((size_t)od*T); if(ok)ggml_backend_tensor_get(h,sc.data(),0,ggml_nbytes(h));
  ggml_gallocr_free(ga);ggml_free(c);ggml_backend_free(be);if(m.ctx)ggml_free(m.ctx);
  if(!ok)return false;

  // ===== E2EVadModel state machine (host) -> speech segments [start_ms,end_ms] =====
  const int FR=10;                 // ms per frame (frame_in_ms)
  const int win=20;                // window_size_ms 200 / 10
  const int s2s=15, sp2s=15;       // sil_to_speech / speech_to_sil thres (150/10)
  const int lookahead_end=100/FR;  // lookahead_time_end_point 100/10 = 10 (do_extend)
  int max_seg = (max_seg_ms>0 ? max_seg_ms : 60000)/FR;       // max_single_segment frames
  const int start_lookback = win + 200/FR;                   // LatencyFrmNumAtStartPoint = 40
  // End-silence threshold from DEFAULT_SILENCE_SCHEDULE, stepped at chunk boundaries (chunk_size
  // 60000ms = 6000 frames). At each boundary, if mid-segment (InSpeech) or in_speech latched,
  // accumulated_ms += 60000; threshold = schedule(accumulated). Reset to 0 on each segment emit.
  const int CHUNK=60000/FR;
  int acc=0, insp=0; int max_end_sil, end_lookback;
  auto recompute=[&](){
    int s; if(acc<=10000)s=2000; else if(acc<=20000)s=1000; else if(acc<=30000)s=800;
    else if(acc<=40000)s=600; else if(acc<=50000)s=400; else if(acc<=60000)s=200; else s=100;
    int ms=s-150; if(ms<0)ms=0; max_end_sil=ms/FR;
    end_lookback=max_end_sil-lookahead_end-1; if(end_lookback<0)end_lookback=0;
  };
  recompute();
  std::vector<int> wbuf(win,0); int wpos=0,wsum=0,pre=0;
  int st=0, cstart=-1, csil=0, prev_end=0;
  auto reset=[&](){ std::fill(wbuf.begin(),wbuf.end(),0); wpos=0; wsum=0; pre=0; csil=0; st=0; cstart=-1; acc=0; insp=0; };
  auto emit=[&](int s,int e){ if(s<prev_end)s=prev_end; if(s<0)s=0; if(e>T)e=T; if(e>s){segs.push_back({s,e}); prev_end=e;} };
  for(int t=0;t<T;t++){
    if(t>0 && t%CHUNK==0){ if(st==1||insp){acc+=60000; insp=1;} recompute(); }
    float sil=sc[(size_t)t*od+0];
    int fs = ((1.0f-sil) >= sil + 0.5f) ? 1 : 0;               // speech_noise_thres=0.5
    wsum -= wbuf[wpos]; wsum += fs; wbuf[wpos]=fs; wpos=(wpos+1)%win;
    int ch;
    if(pre==0 && wsum>=s2s){pre=1; ch=3;}
    else if(pre==1 && wsum<=sp2s){pre=0; ch=1;}
    else ch = pre==0?0:2;
    if(ch==3){ csil=0;
      if(st==0){ cstart=t-start_lookback; if(cstart<prev_end)cstart=prev_end; if(cstart<0)cstart=0; st=1; }
      else if(st==1 && t-cstart+1>max_seg){ emit(cstart,t); reset(); }
    } else if(ch==1||ch==2){ csil=0;
      if(st==1 && t-cstart+1>max_seg){ emit(cstart,t); reset(); }
    } else { csil++;
      if(st==1){
        if(csil>=max_end_sil){ emit(cstart, t-end_lookback); reset(); }
        else if(t-cstart+1>max_seg){ emit(cstart,t); reset(); }
      }
    }
  }
  if(st==1) emit(cstart,T);
  // convert frame indices -> ms
  for(auto&s:segs){ s.first*=FR; s.second*=FR; }
  return true;
}
