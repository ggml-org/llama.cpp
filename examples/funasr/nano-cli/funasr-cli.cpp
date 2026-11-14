// funasr-cli: end-to-end Fun-ASR-Nano in C++ on the llama.cpp / ggml stack.
//
//   wav(16k mono) -> kaldi fbank -> SAN-M encoder + adaptor (ggml) ->
//   low-frame-rate truncation -> [prefix tokens | audio embeds | suffix tokens]
//   -> Qwen3 LLM (llama.cpp) -> transcription.
//
// This is the whisper.cpp-style single-binary path: no Python at runtime.
//
// NOTE: this source contains one non-ASCII string, the Chinese transcript prompt
// "语音转写：" ("transcribe:") - it is required verbatim by the trained model.
// Everything else in the file is ASCII.
//
//   funasr-cli --enc funasr-encoder.gguf -m qwen3-0.6b.gguf -a audio.wav

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // not guaranteed by <cmath> on MSVC
#endif

// any audio (wav/mp3/flac, any rate/channels) -> 16 kHz mono f32, via miniaudio
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
// built-in FSMN-VAD front end (single-binary --vad segmentation)
#include "funasr_vad.h"
#include <utility>

// ======================= kaldi fbank + LFR =======================
static const int FS=16000, WINLEN=400, SHIFT=160, NFFT=512, NMEL=80, LFR_M=7, LFR_N=6;
static const float PREEMPH=0.97f, LOWF=20.0f, HIGHF=8000.0f;
static inline float mel(float f){ return 1127.0f*logf(1.0f+f/700.0f); }
static void fft(std::vector<float>&re,std::vector<float>&im,int n){
    for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
    for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
        for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){
            float ur=re[i+k],ui=im[i+k];float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
            re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;
            float n2=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=n2;}}}
}
// returns [T x 560] row-major, sets T
static std::vector<float> compute_fbank(std::vector<float> wav, int & T_out) {
    for (auto & v : wav) v *= 32768.0f;
    std::vector<float> win(WINLEN);
    for (int i=0;i<WINLEN;i++) win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
    const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT, ml=mel(LOWF), mh=mel(HIGHF), dm=(mh-ml)/(NMEL+1);
    std::vector<std::vector<float>> fb(NMEL, std::vector<float>(NBIN,0.0f));
    for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
        for(int k=0;k<NBIN;k++){float mf=mel(bw*k); if(mf>L&&mf<R) fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
    int N=wav.size(); int T=(N-WINLEN)/SHIFT+1;
    std::vector<std::vector<float>> feat(T, std::vector<float>(NMEL));
    std::vector<float> re(NFFT),im(NFFT),fr(WINLEN);
    const float fl=1.1920929e-07f;
    for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;
        double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
        for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;
        for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
        for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}
        fft(re,im,NFFT);
        for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);
            feat[t][m]=logf(e>fl?e:fl);}}
    // LFR
    const int pad=(LFR_M-1)/2; int T_lfr=(T+LFR_N-1)/LFR_N;
    std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
    for(int i=0;i<pad;i++)pd.push_back(feat[0]);
    for(int t=0;t<T;t++)pd.push_back(feat[t]);
    while((int)pd.size()<(T_lfr-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
    int D=LFR_M*NMEL; std::vector<float> out((size_t)T_lfr*D);
    for(int i=0;i<T_lfr;i++)for(int j=0;j<LFR_M;j++)
        memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
    T_out=T_lfr; return out;
}

// ======================= ggml SAN-M encoder + adaptor =======================
struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,adp_llm=1024,adp_layers=2,adp_head=8; };
struct enc_model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
    // CPU byte copies of the weights (fixed at load_enc time); run_encoder uploads each
    // segment from these, so the source data is not dangling after tensor->data is
    // repointed at a GPU buffer.
    std::vector<std::pair<struct ggml_tensor*,std::vector<uint8_t>>> cpu_w;
    ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };
static const float LN_EPS=1e-5f;
static bool load_enc(const char*p, enc_model&m){
    gguf_init_params gp={false,&m.ctx_w}; gguf_context*g=gguf_init_from_file(p,gp); if(!g)return false;
    auto rd=[&](const char*k,int d){int i=gguf_find_key(g,k);return i<0?d:(int)gguf_get_val_u32(g,i);};
    m.c.d_model=rd("funasr.enc.output_size",512); m.c.n_head=rd("funasr.enc.attention_heads",4);
    m.c.num_blocks=rd("funasr.enc.num_blocks",50); m.c.tp_blocks=rd("funasr.enc.tp_blocks",20);
    m.c.kernel=rd("funasr.enc.kernel_size",11); m.c.adp_llm=rd("funasr.adp.llm_dim",1024);
    m.c.adp_layers=rd("funasr.adp.n_layer",2); m.c.adp_head=rd("funasr.adp.attention_heads",8);
    int n=gguf_get_n_tensors(g); for(int i=0;i<n;i++){const char*nm=gguf_get_tensor_name(g,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
    for(struct ggml_tensor*ct=ggml_get_first_tensor(m.ctx_w);ct;ct=ggml_get_next_tensor(m.ctx_w,ct)){
        size_t nb=ggml_nbytes(ct); std::vector<uint8_t> buf(nb);
        if(ct->data) memcpy(buf.data(),ct->data,nb);
        m.cpu_w.push_back({ct,std::move(buf)});
    }
    gguf_free(g); return true;
}
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
    ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
    ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
    ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
    ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
    const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
    ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
    for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
        auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
    q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
    ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
    ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T,bool res){
    auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static ggml_tensor* adp_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.adp_llm,H=m.c.adp_head,dk=D/H; auto r=x;
    auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto q=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_q.weight"),m.g(p+"self_attn.linear_q.bias"),h),dk,H,T),0,2,1,3);
    auto k=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_k.weight"),m.g(p+"self_attn.linear_k.bias"),h),dk,H,T),0,2,1,3);
    auto vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_v.weight"),m.g(p+"self_attn.linear_v.bias"),h),dk,H,T),1,2,0,3));
    auto kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    auto o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    x=ggml_add(c,r,lin(c,m.g(p+"self_attn.linear_out.weight"),m.g(p+"self_attn.linear_out.bias"),o)); r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
    double inc=log(10000.0)/(depth/2.0-1.0);
    for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
        x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}
// find a GPU backend device by name ("cuda"/"vulkan"/...); returns nullptr if absent
static bool icontains(const std::string&s,const std::string&needle){
    if(needle.size()>s.size()) return false;
    for(size_t i=0;i+needle.size()<=s.size();i++){
        size_t j=0; for(;j<needle.size();j++){
            if(std::tolower((unsigned char)s[i+j])!=std::tolower((unsigned char)needle[j])) break;
        }
        if(j==needle.size()) return true;
    }
    return false;
}
static ggml_backend_dev_t find_gpu_backend_device(const std::string&backend_name){
    ggml_backend_load_all();
    for(size_t i=0;i<ggml_backend_dev_count();i++){
        ggml_backend_dev_t dev=ggml_backend_dev_get(i);
        if(ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        std::string reg=ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        std::string dev_name=ggml_backend_dev_name(dev);
        if(icontains(reg,backend_name)||icontains(dev_name,backend_name)){
            return dev;
        }
    }
    return nullptr;
}

// fbank [T x F] -> adaptor out [T x adp_llm] row-major
static std::vector<float> run_encoder(enc_model&m,std::vector<float> fbank,int T,int F,int&Dout){
    float sc=sqrtf((float)m.c.d_model); for(auto&v:fbank)v*=sc; add_posenc(fbank,T,F);
    // prefer CUDA backend if available; fall back to CPU
    ggml_backend_dev_t gpu_dev=find_gpu_backend_device("cuda");
    ggml_backend_t be=nullptr;
    if(gpu_dev){ be=ggml_backend_dev_init(gpu_dev,nullptr); if(be){ fprintf(stderr,"[encoder] using CUDA backend\n"); } }
    if(!be){ be=ggml_backend_cpu_init(); fprintf(stderr,"[encoder] using CPU backend\n"); }
    if(be){
        size_t vfree=0,vtotal=0;
        if(gpu_dev) ggml_backend_dev_memory(gpu_dev,&vfree,&vtotal);
        fprintf(stderr,"[encoder] backend=%s VRAM free=%.1fMB total=%.1fMB\n",
            ggml_backend_name(be), vfree/1048576.0, vtotal/1048576.0);
    }
    ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
    // copy encoder weights from CPU ctx into a GPU buffer (only when running on GPU)
    ggml_backend_buffer_t wbuf=nullptr;
    if(gpu_dev&&be){
        ggml_backend_buffer_t probe=ggml_backend_alloc_buffer(be,4096);
        size_t total=0; for(struct ggml_tensor*t=ggml_get_first_tensor(m.ctx_w);t;t=ggml_get_next_tensor(m.ctx_w,t)) total+=ggml_backend_buffer_get_alloc_size(probe,t);
        ggml_backend_buffer_free(probe);
        wbuf=ggml_backend_alloc_buffer(be,total);
        char* base=(char*)ggml_backend_buffer_get_base(wbuf); size_t off=0;
        fprintf(stderr,"[encoder] weight buffer %zu bytes on GPU\n",total);
        // source data comes from the m.cpu_w CPU copies, so tensor->data repointed at
        // a GPU buffer in a previous segment is not read from while dangling
        for(auto&cw:m.cpu_w){
            struct ggml_tensor*t=cw.first; size_t n=ggml_nbytes(t);
            t->buffer=wbuf; t->data=base+off; off+=ggml_backend_buffer_get_alloc_size(wbuf,t);
            ggml_backend_tensor_set(t,cw.second.data(),0,n);
        }
    }
    ggml_tensor*inp=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,T); ggml_set_input(inp);
    ggml_tensor*x=sanm_layer(c,m,"audio_encoder.encoders0.0.",inp,T,false);
    for(int i=0;i<m.c.num_blocks-1;i++) x=sanm_layer(c,m,"audio_encoder.encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.after_norm.weight"),m.g("audio_encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) x=sanm_layer(c,m,"audio_encoder.tp_encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.tp_norm.weight"),m.g("audio_encoder.tp_norm.bias"));
    x=lin(c,m.g("audio_adaptor.linear1.weight"),m.g("audio_adaptor.linear1.bias"),x); x=ggml_relu(c,x);
    x=lin(c,m.g("audio_adaptor.linear2.weight"),m.g("audio_adaptor.linear2.bias"),x);
    for(int i=0;i<m.c.adp_layers;i++) x=adp_layer(c,m,"audio_adaptor.blocks."+std::to_string(i)+".",x,T);
    ggml_set_output(x);
    ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(gf,x);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga,gf);
    ggml_backend_tensor_set(inp,fbank.data(),0,ggml_nbytes(inp));
    if(!gpu_dev) ggml_backend_cpu_set_n_threads(be,8);
    ggml_backend_graph_compute(be,gf);
    Dout=(int)x->ne[0]; std::vector<float> out((size_t)Dout*T); ggml_backend_tensor_get(x,out.data(),0,ggml_nbytes(x));
    ggml_gallocr_free(ga); ggml_free(c);
    if(wbuf) ggml_backend_buffer_free(wbuf);
    ggml_backend_free(be); return out;
}

// ======================= LLM (llama.cpp) =======================
static int decode_batch(llama_context*ctx,int n,llama_token*tok,float*embd,int n_embd,int&n_past,bool last_logits){
    std::vector<llama_pos> pos(n); std::vector<int32_t> nsid(n,1);
    std::vector<llama_seq_id> s0(1,0); std::vector<llama_seq_id*> sid(n); std::vector<int8_t> lg(n,0);
    for(int i=0;i<n;i++){pos[i]=n_past+i;sid[i]=s0.data();}
    if(last_logits) lg[n-1]=1;
    llama_batch b={n,tok,embd,pos.data(),nsid.data(),sid.data(),lg.data()};
    int r=llama_decode(ctx,b); n_past+=n; return r;
}

int main(int argc,char**argv){
    static int64_t _t0log=0;
    auto LOG=[&](const char*fmt,...){
        char b[1024]; va_list ap; va_start(ap,fmt); vsnprintf(b,sizeof(b),fmt,ap); va_end(ap);
        int64_t us=ggml_time_us(); double s=_t0log?((double)us-_t0log)/1e6:0.0;
        fprintf(stderr,"[%8.2fs] %s\n",s,b); fflush(stderr);
    };
    std::string enc_path,llm_path,wav_path,vad_path; int npred=512; double chunk_sec=0; float rep=1.0f;
    int vad_maxseg=30000;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--enc")&&i+1<argc)enc_path=argv[++i];
        else if(!strcmp(argv[i],"-m")&&i+1<argc)llm_path=argv[++i];
        else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)npred=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")&&i+1<argc)chunk_sec=atof(argv[++i]);
        else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
        else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--rep")&&i+1<argc)rep=atof(argv[++i]);
        else {fprintf(stderr,"usage: %s --enc enc.gguf -m llm.gguf -a audio.wav [-n npred] [--chunk sec] [--vad fsmn-vad.gguf [--vad-maxseg ms]]\n",argv[0]);return 1;}
    }
    if(enc_path.empty()||llm_path.empty()||wav_path.empty()){fprintf(stderr,"missing args\n");return 1;}
    ggml_time_init();
    _t0log=ggml_time_us();
    LOG("start enc=%s llm=%s wav=%s vad=%s npred=%d chunk=%.0fs vad_maxseg=%dms",
        enc_path.c_str(),llm_path.c_str(),wav_path.c_str(),vad_path.empty()?"-":vad_path.c_str(),npred,chunk_sec,vad_maxseg);

    std::vector<float> wav;
    if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){LOG("failed to read audio");return 1;}
    LOG("audio loaded %.1f sec (%zu samples)", (double)wav.size()/16000.0, wav.size());
    int64_t t0=ggml_time_us();

    enc_model em; if(!load_enc(enc_path.c_str(),em)){LOG("failed to load encoder");return 1;}
    LOG("encoder loaded: d_model=%d blocks=%d tp_blocks=%d kernel=%d adp_llm=%d adp_layers=%d",
        em.c.d_model,em.c.num_blocks,em.c.tp_blocks,em.c.kernel,em.c.adp_llm,em.c.adp_layers);
    ggml_backend_load_all();
    // n_gpu_layers=999: offload every layer to the GPU backend (CUDA when available,
    // falls back to CPU automatically). The value is larger than any real model's
    // layer count, so effectively means "all layers".
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=999;
    llama_model*model=llama_model_load_from_file(llm_path.c_str(),mp); if(!model){LOG("failed to load LLM model");return 1;}
    LOG("LLM model loaded");
    const llama_vocab*vocab=llama_model_get_vocab(model);
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;
    llama_context*ctx=llama_init_from_model(model,cp);
    if(!ctx){LOG("failed to create llama context");llama_model_free(model);return 1;}
    LOG("llama context ready");
    auto sp=llama_sampler_chain_default_params(); llama_sampler*smpl=llama_sampler_chain_init(sp);
    if(rep!=1.0f) llama_sampler_chain_add(smpl,llama_sampler_init_penalties(llama_vocab_n_tokens(vocab),256,rep,0.0f,0.0f));
    llama_sampler_chain_add(smpl,llama_sampler_init_greedy());

    // the transcript prompt is Chinese ("语音转写：" = "transcribe:"): the model was
    // trained with this exact prompt, so the non-ASCII bytes are required here
    const char*prefix="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：";
    const char*suffix="<|im_end|>\n<|im_start|>assistant\n";
    auto tokenize=[&](const char*s){int n=-llama_tokenize(vocab,s,strlen(s),nullptr,0,false,true);
        std::vector<llama_token> v(n); llama_tokenize(vocab,s,strlen(s),v.data(),n,false,true); return v;};
    auto pre=tokenize(prefix); auto suf=tokenize(suffix);

    // Build the list of [offset,len] windows to transcribe (in samples).
    //   --vad : FSMN-VAD speech segments (single-binary front end, replaces fixed chunking)
    //   --chunk sec : fixed-size chunks ; otherwise the whole file in one window
    std::vector<std::pair<int,int>> wins;   // {sample offset, sample len}
    if(!vad_path.empty()){
        std::vector<std::pair<int,int>> segs; // ms
        if(!funasr_vad_segments(vad_path,wav,vad_maxseg,segs)){LOG("vad failed");return 1;}
        for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
            if(end>(int)wav.size())end=wav.size(); if(end-off>0) wins.push_back({off,end-off}); }
        LOG("vad segments=%zu audio=%.1fs", wins.size(), (double)wav.size()/16000.0);
        for(size_t i=0;i<wins.size();i++)
            LOG("  seg[%zu/%zu] %.2fs-%.2fs (len=%.2fs)", i+1, wins.size(),
                (double)wins[i].first/16000.0, (double)(wins[i].first+wins[i].second)/16000.0,
                (double)wins[i].second/16000.0);
    } else {
        int chunk_n = chunk_sec > 0 ? std::max(1, (int)(chunk_sec*16000)) : (int)wav.size();
        for(size_t off=0; off<wav.size(); off+=chunk_n) wins.push_back({(int)off,(int)std::min((size_t)chunk_n,wav.size()-off)});
        LOG("fixed chunks=%zu chunk=%.1fs", wins.size(), chunk_sec>0?chunk_sec:(double)wav.size()/16000.0);
    }
    std::string full;
    for (size_t wi=0; wi<wins.size(); wi++) {
        auto& w = wins[wi];
        int off = w.first, len = w.second;
        if (len < WINLEN) { LOG("  skip seg[%zu/%zu] len=%d < WINLEN", wi+1, wins.size(), len); continue; }
        int64_t st=ggml_time_us();
        std::vector<float> seg(wav.begin()+off, wav.begin()+off+len);
        int T=0; auto fbank=compute_fbank(seg,T);
        int D=0; auto adp=run_encoder(em,fbank,T,560,D);
        int64_t st2=ggml_time_us();
        int ol=1+(T-3+2)/2; ol=1+(ol-3+2)/2; int n_aud=(ol-1)/2+1;
        LOG("  transcribe seg[%zu/%zu] len=%.1fs T=%d adp=%d n_aud=%d enc=%.2fs",
            wi+1, wins.size(), (double)len/16000.0, T, D, n_aud, (st2-st)/1e6);

        llama_memory_clear(llama_get_memory(ctx), true);  // fresh context per chunk
        int n_past=0;
        int rc=decode_batch(ctx,pre.size(),pre.data(),nullptr,0,n_past,false);
        if(rc) { LOG("  [err] llama_decode(prefix) rc=%d at seg[%zu/%zu]", rc, wi+1, wins.size()); break; }
        rc=decode_batch(ctx,n_aud,nullptr,adp.data(),D,n_past,false);
        if(rc) { LOG("  [err] llama_decode(audio) rc=%d at seg[%zu/%zu]", rc, wi+1, wins.size()); break; }
        rc=decode_batch(ctx,suf.size(),suf.data(),nullptr,0,n_past,true);
        if(rc) { LOG("  [err] llama_decode(suffix) rc=%d at seg[%zu/%zu]", rc, wi+1, wins.size()); break; }
        llama_token tk=llama_sampler_sample(smpl,ctx,-1);
        int gen=0;
        for(int i=0;i<npred;i++){
            if(llama_vocab_is_eog(vocab,tk))break;
            char buf[256]; int k=llama_token_to_piece(vocab,tk,buf,sizeof(buf),0,true);
            if(k>0) full.append(buf,k);
            gen++;
            rc=decode_batch(ctx,1,&tk,nullptr,0,n_past,true);
            if(rc) { LOG("  [err] llama_decode(token) rc=%d at seg[%zu/%zu] tok=%d", rc, wi+1, wins.size(), gen); break; }
            tk=llama_sampler_sample(smpl,ctx,-1);
        }
        LOG("  seg[%zu/%zu] done gen_tokens=%d eog=%s total=%.2fs full_chars=%zu",
            wi+1, wins.size(), gen, llama_vocab_is_eog(vocab,tk)?"yes":"no (truncated)", (ggml_time_us()-st)/1e6, full.size());
    }
    printf("%s\n", full.c_str());
    int64_t t2=ggml_time_us();
    LOG("done %.2fs ; chunk=%.0fs ; chars=%zu", (t2-t0)/1e6, chunk_sec, full.size());
    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    if(em.ctx_w) ggml_free(em.ctx_w);
    return 0;
}
