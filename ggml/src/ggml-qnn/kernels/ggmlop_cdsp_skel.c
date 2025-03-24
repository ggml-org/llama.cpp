#include <string.h>
#ifndef _WIN32
#include "HAP_farf.h"
#endif //_WIN32 for HAP_farf
#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H

#include <stdlib.h>
#include <stdint.h>
#include "version_note.h"
#include "ggmlop_ap_skel.h"

typedef struct _heap _heap;
struct _heap {
   _heap* pPrev;
   const char* loc;
   uint64_t buf;
};

typedef struct _allocator {
   _heap* pheap;
   uint8_t* stack;
   uint8_t* stackEnd;
   int nSize;
} _allocator;

_ATTRIBUTE_UNUSED
static __inline int _heap_alloc(_heap** ppa, const char* loc, int size, void** ppbuf) {
   _heap* pn = 0;
   pn = MALLOC((size_t)size + sizeof(_heap) - sizeof(uint64_t));
   if(pn != 0) {
      pn->pPrev = *ppa;
      pn->loc = loc;
      *ppa = pn;
      *ppbuf = (void*)&(pn->buf);
      return 0;
   } else {
      return -1;
   }
}
#define _ALIGN_SIZE(x, y) (((x) + (y-1)) & ~(y-1))

_ATTRIBUTE_UNUSED
static __inline int _allocator_alloc(_allocator* me,
                                    const char* loc,
                                    int size,
                                    unsigned int al,
                                    void** ppbuf) {
   if(size < 0) {
      return -1;
   } else if (size == 0) {
      *ppbuf = 0;
      return 0;
   }
   if((_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + (size_t)size) < (uintptr_t)me->stack + (size_t)me->nSize) {
      *ppbuf = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al);
      me->stackEnd = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + size;
      return 0;
   } else {
      return _heap_alloc(&me->pheap, loc, size, ppbuf);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_deinit(_allocator* me) {
   _heap* pa = me->pheap;
   while(pa != 0) {
      _heap* pn = pa;
      const char* loc = pn->loc;
      (void)loc;
      pa = pn->pPrev;
      FREE(pn);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_init(_allocator* me, uint8_t* stack, int stackSize) {
   me->stack =  stack;
   me->stackEnd =  stack + stackSize;
   me->nSize = stackSize;
   me->pheap = 0;
}


#endif // _ALLOCATOR_H

#ifndef SLIM_H
#define SLIM_H

#include <stdint.h>

//a C data structure for the idl types that can be used to implement
//static and dynamic language bindings fairly efficiently.
//
//the goal is to have a minimal ROM and RAM footprint and without
//doing too many allocations.  A good way to package these things seemed
//like the module boundary, so all the idls within  one module can share
//all the type references.


#define PARAMETER_IN       0x0
#define PARAMETER_OUT      0x1
#define PARAMETER_INOUT    0x2
#define PARAMETER_ROUT     0x3
#define PARAMETER_INROUT   0x4

//the types that we get from idl
#define TYPE_OBJECT             0x0
#define TYPE_INTERFACE          0x1
#define TYPE_PRIMITIVE          0x2
#define TYPE_ENUM               0x3
#define TYPE_STRING             0x4
#define TYPE_WSTRING            0x5
#define TYPE_STRUCTURE          0x6
#define TYPE_UNION              0x7
#define TYPE_ARRAY              0x8
#define TYPE_SEQUENCE           0x9

//these require the pack/unpack to recurse
//so it's a hint to those languages that can optimize in cases where
//recursion isn't necessary.
#define TYPE_COMPLEX_STRUCTURE  (0x10 | TYPE_STRUCTURE)
#define TYPE_COMPLEX_UNION      (0x10 | TYPE_UNION)
#define TYPE_COMPLEX_ARRAY      (0x10 | TYPE_ARRAY)
#define TYPE_COMPLEX_SEQUENCE   (0x10 | TYPE_SEQUENCE)


typedef struct Type Type;

#define INHERIT_TYPE\
   int32_t nativeSize;                /*in the simple case its the same as wire size and alignment*/\
   union {\
      struct {\
         const uintptr_t         p1;\
         const uintptr_t         p2;\
      } _cast;\
      struct {\
         uint32_t  iid;\
         uint32_t  bNotNil;\
      } object;\
      struct {\
         const Type  *arrayType;\
         int32_t      nItems;\
      } array;\
      struct {\
         const Type *seqType;\
         int32_t      nMaxLen;\
      } seqSimple; \
      struct {\
         uint32_t bFloating;\
         uint32_t bSigned;\
      } prim; \
      const SequenceType* seqComplex;\
      const UnionType  *unionType;\
      const StructType *structType;\
      int32_t         stringMaxLen;\
      uint8_t        bInterfaceNotNil;\
   } param;\
   uint8_t    type;\
   uint8_t    nativeAlignment\

typedef struct UnionType UnionType;
typedef struct StructType StructType;
typedef struct SequenceType SequenceType;
struct Type {
   INHERIT_TYPE;
};

struct SequenceType {
   const Type *         seqType;
   uint32_t               nMaxLen;
   uint32_t               inSize;
   uint32_t               routSizePrimIn;
   uint32_t               routSizePrimROut;
};

//byte offset from the start of the case values for
//this unions case value array.  it MUST be aligned
//at the alignment requrements for the descriptor
//
//if negative it means that the unions cases are
//simple enumerators, so the value read from the descriptor
//can be used directly to find the correct case
typedef union CaseValuePtr CaseValuePtr;
union CaseValuePtr {
   const uint8_t*   value8s;
   const uint16_t*  value16s;
   const uint32_t*  value32s;
   const uint64_t*  value64s;
};

//these are only used in complex cases
//so I pulled them out of the type definition as references to make
//the type smaller
struct UnionType {
   const Type           *descriptor;
   uint32_t               nCases;
   const CaseValuePtr   caseValues;
   const Type * const   *cases;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
   uint8_t                inCaseAlignment;
   uint8_t                routCaseAlignmentPrimIn;
   uint8_t                routCaseAlignmentPrimROut;
   uint8_t                nativeCaseAlignment;
   uint8_t              bDefaultCase;
};

struct StructType {
   uint32_t               nMembers;
   const Type * const   *members;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
};

typedef struct Parameter Parameter;
struct Parameter {
   INHERIT_TYPE;
   uint8_t    mode;
   uint8_t  bNotNil;
};

#define SLIM_IFPTR32(is32,is64) (sizeof(uintptr_t) == 4 ? (is32) : (is64))
#define SLIM_SCALARS_IS_DYNAMIC(u) (((u) & 0x00ffffff) == 0x00ffffff)

typedef struct Method Method;
struct Method {
   uint32_t                    uScalars;            //no method index
   int32_t                     primInSize;
   int32_t                     primROutSize;
   int                         maxArgs;
   int                         numParams;
   const Parameter * const     *params;
   uint8_t                       primInAlignment;
   uint8_t                       primROutAlignment;
};

typedef struct Interface Interface;

struct Interface {
   int                            nMethods;
   const Method  * const          *methodArray;
   int                            nIIds;
   const uint32_t                   *iids;
   const uint16_t*                  methodStringArray;
   const uint16_t*                  methodStrings;
   const char*                    strings;
};


#endif //SLIM_H


#ifndef _GGMLOP_SLIM_H
#define _GGMLOP_SLIM_H
#include <stdint.h>

#ifndef __QAIC_SLIM
#define __QAIC_SLIM(ff) ff
#endif
#ifndef __QAIC_SLIM_EXPORT
#define __QAIC_SLIM_EXPORT
#endif

static const Type types[5];
static const Type* const typeArrays[6] = {&(types[0]),&(types[1]),&(types[1]),&(types[0]),&(types[0]),&(types[3])};
static const StructType structTypes[1] = {{0x6,&(typeArrays[0]),0x58,0x4,0x50,0x8,0x4,0x8}};
static const Type types[5] = {{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4},{0x20,{{(const uintptr_t)&(types[2]),(const uintptr_t)0x4}}, 8,0x8},{0x8,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x8},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[4]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8)},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4}};
static const Parameter parameters[5] = {{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x58,0x60),{{(const uintptr_t)&(structTypes[0]),0}}, 22,0x8,0,0},{SLIM_IFPTR32(0x58,0x60),{{(const uintptr_t)&(structTypes[0]),0}}, 22,0x8,3,0}};
static const Parameter* const parameterArrays[6] = {(&(parameters[3])),(&(parameters[3])),(&(parameters[4])),(&(parameters[0])),(&(parameters[1])),(&(parameters[2]))};
static const Method methods[3] = {{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x1),0x4,0x0,2,2,(&(parameterArrays[3])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x0,0x1,0x0),0x0,0x0,1,1,(&(parameterArrays[5])),0x1,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x2,0x0,0x0),0xb4,0x50,3,3,(&(parameterArrays[0])),0x8,0x8}};
static const Method* const methodArrays[4] = {&(methods[0]),&(methods[1]),&(methods[2]),&(methods[2])};
static const char strings[68] = "mulmat\0flags\0close\0src1\0data\0type\0src0\0open\0dst\0add\0uri\0op\0nb\0ne\0h\0";
static const uint16_t methodStrings[49] = {0,34,29,62,59,56,7,24,19,29,62,59,56,7,24,44,29,62,59,56,7,24,48,34,29,62,59,56,7,24,19,29,62,59,56,7,24,44,29,62,59,56,7,24,39,52,65,13,65};
static const uint16_t methodStringsArrays[4] = {44,47,22,0};
__QAIC_SLIM_EXPORT const Interface __QAIC_SLIM(ggmlop_slim) = {4,&(methodArrays[0]),0,0,&(methodStringsArrays [0]),methodStrings,strings};
#endif //_GGMLOP_SLIM_H
extern int adsp_mmap_fd_getinfo(int, uint32_t *);
#ifdef __cplusplus
extern "C" {
#endif
_ATTRIBUTE_VISIBILITY uint32_t ggmlop_skel_handle_invoke_qaic_version = 10048;
_ATTRIBUTE_VISIBILITY char ggmlop_skel_handle_invoke_uri[77+1]="file:///libggmlop_skel.so?ggmlop_skel_handle_invoke&_modver=1.0&_idlver=0.0.1";
static __inline int _skel_pack(_ATTRIBUTE_UNUSED remote_arg* _praROutPost, _ATTRIBUTE_UNUSED remote_arg* _ppraROutPost[1], _ATTRIBUTE_UNUSED void* _primROut, _ATTRIBUTE_UNUSED uint32_t _rout0[1], _ATTRIBUTE_UNUSED uint64_t _rout1[4], _ATTRIBUTE_UNUSED uint64_t _rout2[4], _ATTRIBUTE_UNUSED uint32_t _rout3[1], _ATTRIBUTE_UNUSED uint32_t _rout4[1], _ATTRIBUTE_UNUSED char* _rout5[1], _ATTRIBUTE_UNUSED uint32_t _rout5Len[1]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _COPY(_primROut, 0, _rout0, 0, 4);
   _COPY(_primROut, 8, _rout1, 0, 32);
   _COPY(_primROut, 40, _rout2, 0, 32);
   _COPY(_primROut, 72, _rout3, 0, 4);
   _COPY(_primROut, 76, _rout4, 0, 4);
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +1;
   return _nErr;
}
static __inline int _skel_unpack(_ATTRIBUTE_UNUSED _allocator* _al, _ATTRIBUTE_UNUSED remote_arg* _praIn, _ATTRIBUTE_UNUSED remote_arg* _ppraIn[1], _ATTRIBUTE_UNUSED remote_arg* _praROut, _ATTRIBUTE_UNUSED remote_arg* _ppraROut[1], _ATTRIBUTE_UNUSED remote_arg* _praHIn, _ATTRIBUTE_UNUSED remote_arg* _ppraHIn[1], _ATTRIBUTE_UNUSED remote_arg* _praHROut, _ATTRIBUTE_UNUSED remote_arg* _ppraHROut[1], _ATTRIBUTE_UNUSED void* _primIn, _ATTRIBUTE_UNUSED void* _primROut, _ATTRIBUTE_UNUSED uint32_t _rout0[1], _ATTRIBUTE_UNUSED uint64_t _rout1[4], _ATTRIBUTE_UNUSED uint64_t _rout2[4], _ATTRIBUTE_UNUSED uint32_t _rout3[1], _ATTRIBUTE_UNUSED uint32_t _rout4[1], _ATTRIBUTE_UNUSED char* _rout5[1], _ATTRIBUTE_UNUSED uint32_t _rout5Len[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_rout5Len, 0, _primIn, 0, 4);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 4)) >= (size_t)(_rout5Len[0]));
   _rout5[0] = _praROut[0].buf.pv;
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +1;
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_unpack_1(_ATTRIBUTE_UNUSED _allocator* _al, _ATTRIBUTE_UNUSED remote_arg* _praIn, _ATTRIBUTE_UNUSED remote_arg* _ppraIn[1], _ATTRIBUTE_UNUSED remote_arg* _praROut, _ATTRIBUTE_UNUSED remote_arg* _ppraROut[1], _ATTRIBUTE_UNUSED remote_arg* _praHIn, _ATTRIBUTE_UNUSED remote_arg* _ppraHIn[1], _ATTRIBUTE_UNUSED remote_arg* _praHROut, _ATTRIBUTE_UNUSED remote_arg* _ppraHROut[1], _ATTRIBUTE_UNUSED void* _primIn, _ATTRIBUTE_UNUSED void* _primROut, _ATTRIBUTE_UNUSED uint32_t _in0[1], _ATTRIBUTE_UNUSED uint64_t _in1[4], _ATTRIBUTE_UNUSED uint64_t _in2[4], _ATTRIBUTE_UNUSED uint32_t _in3[1], _ATTRIBUTE_UNUSED uint32_t _in4[1], _ATTRIBUTE_UNUSED char* _in5[1], _ATTRIBUTE_UNUSED uint32_t _in5Len[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 8, 32);
   _COPY(_in2, 0, _primIn, 40, 32);
   _COPY(_in3, 0, _primIn, 72, 4);
   _COPY(_in4, 0, _primIn, 76, 4);
   _COPY(_in5Len, 0, _primIn, 80, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in5Len[0]));
   _in5[0] = _praIn[0].buf.pv;
   _ppraInStart[0] += (_praIn - _praInStart) + 1;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method(int (*_pfn)(remote_handle64, const dsptensor*, const dsptensor*, dsptensor*), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint64_t _in0[SLIM_IFPTR32(11, 12)] = {0};
   uint64_t _in1[SLIM_IFPTR32(11, 12)] = {0};
   uint64_t _rout2[SLIM_IFPTR32(11, 12)] = {0};
   uint64_t* _primIn= 0;
   int _numIn[1] = {0};
   uint64_t* _primROut= 0;
   int _numInH[1] = {0};
   int _numROut[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   remote_arg* _praROutPost = 0;
   remote_arg** _ppraROutPost = &_praROutPost;
   _allocator _al[1] = {{0}};
   remote_arg** _ppraIn = &_praIn;
   remote_arg** _ppraROut = &_praROut;
   remote_arg* _praHIn = 0;
   remote_arg** _ppraHIn = &_praHIn;
   remote_arg* _praHROut = 0;
   remote_arg** _ppraHROut = &_praHROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)>=1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)>=1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 180);
   _primIn = _pra[0].buf.pv;
   _QAIC_ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 80);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _numInH[0] = REMOTE_SCALARS_INHANDLES(_sc);
   _numROut[0] = REMOTE_SCALARS_OUTBUFS(_sc);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 1);
   _praROutPost = _praROut;
   _allocator_init(_al, 0, 0);
   if(_praHIn == 0)
   {
      _praHIn = ((_praROut + _numROut[0]) + 1);
   }
   if(_praHROut == 0)
      (_praHROut = _praHIn + _numInH[0] + 0);
   _TRY(_nErr, _skel_unpack_1(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 0), 0, (uint32_t*)&(((uint32_t*)_in0)[0]), (uint64_t*)&(((uint64_t*)_in0)[1]), (uint64_t*)&(((uint64_t*)_in0)[5]), (uint32_t*)&(((uint32_t*)_in0)[18]), (uint32_t*)&(((uint32_t*)_in0)[19]), SLIM_IFPTR32((char**)&(((uint32_t*)_in0)[20]), (char**)&(((uint64_t*)_in0)[10])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_in0)[21]), (uint32_t*)&(((uint32_t*)_in0)[22]))));
   _TRY(_nErr, _skel_unpack_1(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 88), 0, (uint32_t*)&(((uint32_t*)_in1)[0]), (uint64_t*)&(((uint64_t*)_in1)[1]), (uint64_t*)&(((uint64_t*)_in1)[5]), (uint32_t*)&(((uint32_t*)_in1)[18]), (uint32_t*)&(((uint32_t*)_in1)[19]), SLIM_IFPTR32((char**)&(((uint32_t*)_in1)[20]), (char**)&(((uint64_t*)_in1)[10])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_in1)[21]), (uint32_t*)&(((uint32_t*)_in1)[22]))));
   _TRY(_nErr, _skel_unpack(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 176), ((char*)_primROut + 0), (uint32_t*)&(((uint32_t*)_rout2)[0]), (uint64_t*)&(((uint64_t*)_rout2)[1]), (uint64_t*)&(((uint64_t*)_rout2)[5]), (uint32_t*)&(((uint32_t*)_rout2)[18]), (uint32_t*)&(((uint32_t*)_rout2)[19]), SLIM_IFPTR32((char**)&(((uint32_t*)_rout2)[20]), (char**)&(((uint64_t*)_rout2)[10])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_rout2)[21]), (uint32_t*)&(((uint32_t*)_rout2)[22]))));
   _TRY(_nErr, _pfn(_h, (const dsptensor*)_in0, (const dsptensor*)_in1, (dsptensor*)_rout2));
   _TRY(_nErr, _skel_pack((_praROutPost + 0), _ppraROutPost, ((char*)_primROut + 0), (uint32_t*)&(((uint32_t*)_rout2)[0]), (uint64_t*)&(((uint64_t*)_rout2)[1]), (uint64_t*)&(((uint64_t*)_rout2)[5]), (uint32_t*)&(((uint32_t*)_rout2)[18]), (uint32_t*)&(((uint32_t*)_rout2)[19]), SLIM_IFPTR32((char**)&(((uint32_t*)_rout2)[20]), (char**)&(((uint64_t*)_rout2)[10])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_rout2)[21]), (uint32_t*)&(((uint32_t*)_rout2)[22]))));
   _QAIC_CATCH(_nErr) {}
   _allocator_deinit(_al);
   return _nErr;
}
static __inline int _skel_method_1(int (*_pfn)(remote_handle64), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   remote_handle64 _in0[1] = {0};
   remote_arg* _praRHandleIn = _pra + REMOTE_SCALARS_INBUFS(_sc) +  REMOTE_SCALARS_OUTBUFS(_sc);
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((0 + 0) + (((1 + 0) + 0) + 0))) <= _praEnd);
   _COPY(_in0, 0, &(_praRHandleIn[0].h64), 0, sizeof(remote_handle64));
   _TRY(_nErr, _pfn((remote_handle64)*_in0));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _compare_versions(char* stub_ver, char* skel_ver, int* result) {
   unsigned long int major_stub = 0, minor_stub = 0, patch_stub = 0;
   unsigned long int major_skel = 0, minor_skel = 0, patch_skel = 0;
   char *saveptr1 = NULL;
   char *token1 = NULL;
   char *saveptr2 = NULL;
   char *token2 = NULL;
   int i=0;
   for (i=0, token1 = strtok_r(stub_ver, ".", &saveptr1); i<3 && token1 != NULL; i++, token1 = strtok_r(NULL, ".", &saveptr1))
   {
      unsigned long int tn = strtoul(token1, NULL,10);
      if( tn > 999)
      {
         *result=-1;
         return 0;
      }
      else
      {
         if(i==0) major_stub=tn;
         if(i==1) minor_stub=tn;
         if(i==2) patch_stub=tn;
      }
   }
   for (i=0, token2 = strtok_r(skel_ver, ".", &saveptr2); i<3 && token2 != NULL; i++, token2 = strtok_r(NULL, ".", &saveptr2))
   {
      unsigned long int tn = strtoul(token2, NULL,10);
      if( tn > 999)
      {
         *result=-1;
         return 0;
      }
      else
      {
         if(i==0) major_skel=tn;
         if(i==1) minor_skel=tn;
         if(i==2) patch_skel=tn;
      }
   }
   if(major_stub<major_skel)
   {
      *result=1;
      return 0;
   }
   else if(major_stub==major_skel)
   {
      if( minor_stub < minor_skel )
      {
         *result=1;
         return 0;
      }
      else if((minor_stub == minor_skel) && (patch_skel>=patch_stub))
      {
         *result=1;
         return 0;
      }
   }
   *result=-1;
   return 0;
}
static __inline int _stub_skel_version_check(char*_in0, int* resVal) {
   int _nErr = 0;
   char* p = strstr(_in0, "_idlver=");
   if(!p)
   {
      *resVal = -1;
      return 0;
   }
   p+=8;
   int i=0,len=0, comVer=0,num_delimit=0, updtInxStub=0, updtInxSkel=0;
   for(i=0;i<strlen(p);i++)
   {
      if(num_delimit>2)
      {
         *resVal = -1;
         return 0;
      }
      if ((p[i]>='0' && p[i]<='9') || (p[i]=='.'))
      {
         len++;
         if(p[i]=='.')
         {
            num_delimit++;
         }
      }
      else if(p[i]=='&')
      {
         break;
      }
      else
      {
         *resVal = -1;
         return 0;
      }
   }
   char* stubVer=(char*)MALLOC(len+1);
   _QAIC_ASSERT(_nErr, stubVer!=NULL);
   for(i=0;i<strlen(p);i++)
   {
      if((p[i]>='0' && p[i]<='9') || (p[i]=='.'))
      {
         stubVer[updtInxStub]=p[i];
         updtInxStub++;
      }
      else if(p[i]=='&')
      {
         break;
      }
   }
   stubVer[len]='\0';
   char* skelVer=(char*)MALLOC(strlen(IDL_VERSION)+1);
   _QAIC_ASSERT(_nErr, skelVer!=NULL);
   for(i=0;i< strlen(IDL_VERSION);i++)
   {
      skelVer[updtInxSkel]=IDL_VERSION[i];
      updtInxSkel++;
   }
   skelVer[strlen(IDL_VERSION)]='\0';
   _TRY(_nErr, _compare_versions(stubVer, skelVer, &comVer));
   *resVal = 0;
   if (comVer==-1)
   {
      *resVal = -1;
   }
   FREE(stubVer);
   FREE(skelVer);
   _QAIC_CATCH(_nErr) {}
   return 0;
}
static __inline int _skel_method_2(int (*_pfn)(const char*, remote_handle64*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   char* _in0[1] = {0};
   uint32_t _in0Len[1] = {0};
   remote_handle64 _rout1[1] = {0};
   uint32_t* _primIn= 0;
   remote_arg* _praRHandleROut = _pra + REMOTE_SCALARS_INBUFS(_sc) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) ;
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==2);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==1);
   _QAIC_ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 1) + 0) + 0))) <= _praEnd);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _QAIC_ASSERT(_nErr, (_in0Len[0] > 0) && (_in0[0][(_in0Len[0] - 1)] == 0));
   int resVal;
   _TRY(_nErr, _stub_skel_version_check(*_in0, &resVal));
   if(resVal==-1)
   {
      return AEE_ESTUBSKELVERMISMATCH;
   }
   _TRY(_nErr, _pfn((const char*)*_in0, (remote_handle64*)_rout1));
   _COPY(&(_praRHandleROut[0].h64), 0, _rout1, 0, sizeof(remote_handle64));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
__QAIC_SKEL_EXPORT int __QAIC_SKEL(ggmlop_skel_handle_invoke)(remote_handle64 _h, uint32_t _sc, remote_arg* _pra) __QAIC_SKEL_ATTRIBUTE {
   switch(REMOTE_SCALARS_METHOD(_sc)){
      case 0:
      return _skel_method_2(__QAIC_IMPL(ggmlop_dsp_open), _sc, _pra);
      case 1:
      return _skel_method_1(__QAIC_IMPL(ggmlop_dsp_close), _sc, _pra);
      case 2:
      return _skel_method(__QAIC_IMPL(ggmlop_dsp_add), _h, _sc, _pra);
      case 3:
      return _skel_method(__QAIC_IMPL(ggmlop_dsp_mulmat), _h, _sc, _pra);
   }
   return AEE_EUNSUPPORTED;
}

/* Library version needs to be added in the name member of note_type structure in below format
 * "lib.ver.1.0.0." + "<library_name>" + ":" + "<version>"
 */
const lib_ver_note_t so_ver __attribute__ ((section (".note.lib.ver")))
        __attribute__ ((visibility ("default"))) = {
                100,
                0,
                0,
                "lib.ver.1.0.0.libggmlop_skel.so:4.5.0",
        };
