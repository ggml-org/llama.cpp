"""Independent CPU oracle. Native GGML TQ2_0 bytes; explicit Q8g32 activations.

This is NOT an MLX bit-exact oracle. It checks format, routing, fusion and numerical
agreement against FP32 or against the same quantized activations independently.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TQ2:
    data: np.ndarray  # uint8 [experts, rows, K/256, 66]
    k: int

    def __post_init__(self):
        if self.k <= 0 or self.k % 256:
            raise ValueError("K must be a positive multiple of 256")
        if self.data.dtype != np.uint8 or self.data.ndim != 4 or self.data.shape[-2:] != (self.k // 256, 66):
            raise ValueError("TQ2 shape must be [E,M,K/256,66] uint8")
        if not self.data.flags.c_contiguous or min(self.data.shape) == 0:
            raise ValueError("nonempty contiguous TQ2 buffer required")
        if not np.isfinite(self.scales()).all():
            raise ValueError("nonfinite TQ2 scale")

    @property
    def experts(self): return self.data.shape[0]
    @property
    def m(self): return self.data.shape[1]

    def scales(self):
        return self.data[..., 64:66].copy().view('<f2').reshape(self.data.shape[:-1]).astype(np.float32)

    def codes(self):
        sh = np.array([0, 2, 4, 6], dtype=np.uint8)
        q = self.data[..., :64].reshape(*self.data.shape[:-1], 2, 1, 32)
        return ((q >> sh.reshape(1, 1, 1, 1, 4, 1)) & 3).astype(np.int8).reshape(
            self.experts, self.m, self.k) - np.int8(1)

    def decode(self):
        return (self.codes().reshape(self.experts, self.m, self.k//256, 256).astype(np.float32)
                * self.scales()[..., None]).reshape(self.experts, self.m, self.k)

    def expert(self, e: int):
        if not 0 <= e < self.experts: raise ValueError("expert out of range")
        return TQ2(self.data[e:e+1].copy(), self.k)


def pack_codes(codes: np.ndarray, scales: np.ndarray) -> TQ2:
    c = np.asarray(codes)
    if c.ndim == 2: c = c[None]
    if c.ndim != 3 or c.shape[-1] % 256 or c.shape[-1] == 0:
        raise ValueError("codes must have shape [E,M,K] with K%256=0")
    if not np.issubdtype(c.dtype, np.integer) or np.any(c < -1) or np.any(c > 2):
        raise ValueError("representable TQ2 codes are -1,0,1,2; native ternary uses -1,0,1")
    e,m,k = c.shape
    d = np.broadcast_to(np.asarray(scales, dtype=np.float32), (e,m,k//256))
    if not np.isfinite(d).all(): raise ValueError("nonfinite scale")
    with np.errstate(over='ignore'):
        dh = d.astype('<f2')
    if not np.isfinite(dh).all(): raise ValueError("scale overflows FP16")
    q = (c.astype(np.int16)+1).astype(np.uint8).reshape(e,m,k//256,2,4,32)
    raw = q[..., 0, :].copy()
    for s in range(1,4): raw |= q[..., s, :] << (2*s)
    out = np.empty((e,m,k//256,66),dtype=np.uint8)
    out[..., :64] = raw.reshape(e,m,k//256,64)
    out[..., 64:66] = dh[..., None].view(np.uint8)
    return TQ2(out,k)


def random_tq2(rng, e, m, k, *, code3=False):
    c = rng.integers(-1, 3 if code3 else 2, size=(e,m,k), dtype=np.int8)
    d = rng.uniform(0.015,0.12,(e,m,k//256)).astype(np.float32)
    # Include all-zero blocks and varying block scales: no row-alpha assumption.
    d[:,::17,0] = 0
    return pack_codes(c,d)


def q8g32(x):
    x = np.asarray(x,dtype=np.float32)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 32:
        raise ValueError("Q8g32 expects [rows,K], K%32=0")
    if not np.isfinite(x).all(): raise ValueError("nonfinite activation")
    b = x.reshape(x.shape[0],-1,32)
    a = np.max(np.abs(b),axis=-1)
    d = np.where(a == 0, np.float32(0), np.maximum(a/np.float32(127),np.float32(np.finfo(np.float32).tiny)))
    v = np.divide(b,d[...,None],out=np.zeros_like(b),where=d[...,None]!=0)
    q = np.clip(np.sign(v)*np.floor(np.abs(v)+np.float32(0.5)),-127,127).astype(np.int8)
    return np.ascontiguousarray(q.reshape(x.shape)),np.ascontiguousarray(d)


def dense(w: TQ2, x, *, q8=False):
    x = np.asarray(x,dtype=np.float32)
    if w.experts != 1 or x.ndim != 2 or x.shape[1] != w.k:
        raise ValueError("dense shape mismatch")
    if not q8: return x @ w.decode()[0].T
    q,ad = q8g32(x)
    c=w.codes()[0]; d=w.scales()[0]
    result=np.zeros((x.shape[0],w.m),np.float32)
    for g in range(w.k//32):
        isum=q[:,g*32:g*32+32].astype(np.int32) @ c[:,g*32:g*32+32].astype(np.int32).T
        result += (ad[:,g,None]*d[None,:,g//8])*isum.astype(np.float32)
    return result


def route(logits, topk=8, *, tie_high=False, ascending=False):
    z=np.asarray(logits,np.float32)
    if z.ndim!=2 or not 1<=topk<=min(8,z.shape[1]) or z.shape[1]>256:
        raise ValueError("router requires [tokens,E<=256], 1<=topk<=8")
    if not np.isfinite(z).all(): raise ValueError("nonfinite router logits")
    # Match the shader's balanced 256-element reduction topology.
    v=np.zeros((len(z),256),np.float32)
    v[:,:z.shape[1]]=np.exp(z-z.max(axis=1,keepdims=True))
    red=v.copy()
    s=128
    while s:
        red[:,:s]+=red[:,s:2*s];s//=2
    probs=v[:,:z.shape[1]]/red[:,0,None]
    ids=np.empty((len(z),topk),np.uint32)
    scores=np.empty((len(z),topk),np.float32)
    ii=np.arange(z.shape[1])
    for t in range(len(z)):
        order=np.lexsort((-ii if tie_high else ii,-probs[t]))[:topk]
        if ascending: order=order[::-1]
        ids[t]=order
        den=np.float32(0)
        for i in order: den=np.float32(den+probs[t,i])
        den=np.float32(den+np.float32(1e-20))
        scores[t]=probs[t,order]/den
    return ids,scores


def bucket(ids, experts):
    ids=np.asarray(ids)
    if ids.ndim!=2 or not np.issubdtype(ids.dtype,np.integer): raise ValueError("integer [tokens,topk] ids required")
    if np.any(ids<0) or np.any(ids>=experts): raise ValueError("invalid expert id")
    flat=ids.reshape(-1)
    sorted_a=np.argsort(flat,kind='stable').astype(np.uint32)
    counts=np.bincount(flat.astype(np.int64),minlength=experts).astype(np.uint32)
    offsets=np.concatenate([np.zeros(1,np.uint32),np.cumsum(counts,dtype=np.uint32)])
    return offsets,sorted_a


def make_jobs(offsets, rows):
    jobs=[]
    for e in range(len(offsets)-1):
        start,end=map(int,offsets[e:e+2])
        for a in range(start,end,4):
            for r in range(0,rows,32): jobs.append((e,a,min(4,end-a),r))
    return np.asarray(jobs,np.uint32).reshape(-1,4)


def job_capacity(assignments,experts,rows):
    if min(assignments,experts,rows)<=0: raise ValueError("positive dimensions required")
    return ((assignments+3*min(assignments,experts))//4)*((rows+31)//32)


def project_routed(w, x, ids, *, per_assignment=False, q8=False):
    ids=np.asarray(ids,np.uint32)
    bucket(ids,w.experts)  # validate
    x=np.asarray(x,np.float32)
    if x.shape != ((ids.size if per_assignment else len(ids)),w.k): raise ValueError("routed shape mismatch")
    out=np.empty((ids.size,w.m),np.float32)
    for e in np.unique(ids):
        a=np.flatnonzero(ids.reshape(-1)==e)
        inp=x[a if per_assignment else a//ids.shape[1]]
        out[a]=dense(w.expert(int(e)),inp,q8=q8)
    return out


def clamped_swiglu(up,gate):
    u=np.clip(np.asarray(up,np.float32),-7,7)
    g=np.minimum(np.asarray(gate,np.float32),np.float32(7))
    # Numerically stable sigmoid, retaining a one-sided gate clamp.
    sig=np.empty_like(g); pos=g>=0
    sig[pos]=np.float32(1)/(np.float32(1)+np.exp(-g[pos]))
    t=np.exp(g[~pos]);sig[~pos]=t/(np.float32(1)+t)
    return (g*sig)*u


def reduce_experts(out,scores):
    s=np.asarray(scores,np.float32)
    o=np.asarray(out,np.float32).reshape(*s.shape,-1)
    y=np.zeros((len(s),o.shape[-1]),np.float32)
    for i in range(s.shape[1]): y+=o[:,i]*s[:,i,None]
    return y
