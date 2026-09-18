#!/usr/bin/env python3
"""Linux-only supplemental shader oracle using Mesa/EGL OpenGL compute.

Transforms only Vulkan binding/push-constant syntax. Q8 native dot is replaced by
four signed integer multiplies for this test. This does NOT compile SPIR-V, test
Vulkan synchronization/host code, measure A750 performance, or validate native DP4A.
"""
from pathlib import Path
import argparse,ctypes as C,json,os,re,shlex,struct,time

U=C.c_uint;I=C.c_int;V=C.c_void_p;S=C.c_ssize_t
class Oracle:
    def __init__(self):
        os.environ['EGL_PLATFORM']='surfaceless';os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
        self.egl=C.CDLL('libEGL.so.1');self.gl=C.CDLL('libGL.so.1')
        def ef(n,ret,args):f=getattr(self.egl,n);f.restype=ret;f.argtypes=args;return f
        self.display=ef('eglGetDisplay',V,[V])(None)
        ma=I();mi=I()
        if not ef('eglInitialize',U,[V,C.POINTER(I),C.POINTER(I)])(self.display,C.byref(ma),C.byref(mi)):raise RuntimeError('EGL initialization failed')
        if not ef('eglBindAPI',U,[U])(0x30A2):raise RuntimeError('eglBindAPI failed')
        attrs=(I*7)(0x3040,8,0x3033,1,0x3024,8,0x3038);config=V();count=I()
        if not ef('eglChooseConfig',U,[V,C.POINTER(I),C.POINTER(V),I,C.POINTER(I)])(self.display,attrs,C.byref(config),1,C.byref(count)) or not count.value:raise RuntimeError('no EGL configuration')
        ca=(I*7)(0x3098,4,0x30FB,5,0x30FD,1,0x3038)
        self.context=ef('eglCreateContext',V,[V,V,V,C.POINTER(I)])(self.display,config,None,ca)
        if not self.context or not ef('eglMakeCurrent',U,[V,V,V,V])(self.display,None,None,self.context):raise RuntimeError('OpenGL 4.5 context unavailable')
        self.functions={};self.buffers={};self.programs={}
        self.renderer=self.fn('glGetString',C.c_char_p,[U])(0x1F01).decode()
        self.version=self.fn('glGetString',C.c_char_p,[U])(0x1F02).decode()
        self.uniform=self.new_buffer(0x8A11,b'\0'*128)
        print('Supplemental software oracle:',self.renderer,self.version,flush=True)
    def fn(self,n,ret,args):
        if n not in self.functions:
            f=getattr(self.gl,n);f.restype=ret;f.argtypes=args;self.functions[n]=f
        return self.functions[n]
    def err(self,label):
        e=self.fn('glGetError',U,[])()
        if e:raise RuntimeError(f'{label}: GL error 0x{e:x}')
    def new_buffer(self,target,data):
        b=U();self.fn('glGenBuffers',None,[I,C.POINTER(U)])(1,C.byref(b));self.fn('glBindBuffer',None,[U,U])(target,b)
        arr=C.create_string_buffer(data);self.fn('glBufferData',None,[U,S,V,U])(target,len(data),arr,0x88E8);self.err('buffer data');return b.value
    def program(self,root,name):
        if name in self.programs:return self.programs[name]
        defs='';is_q8=False
        if name.startswith('project_'):
            m=re.fullmatch(r'project_q([01])_f([012])_r([01])',name)
            if not m:raise ValueError(name)
            q,f,r=m.groups();defs=f'\n#define Q8 {q}\n#define FUSION {f}\n#define REUSE {r}\n';is_q8=q=='1';file=root/'project.comp'
        else:file=root/(name+'.comp')
        text=file.read_text()
        text=re.sub(r'#include\s+"([^"]+)"',lambda m:(root/m.group(1)).read_text(),text)
        text=text.replace('#version 450','#version 450\n'+defs)
        text=re.sub(r'^#extension GL_GOOGLE_include_directive.*$', '',text,flags=re.M)
        text=re.sub(r'^#extension GL_EXT_integer_dot_product.*$', '',text,flags=re.M)
        text=text.replace('set=0,','').replace('layout(push_constant)','layout(std140,binding=0)')
        if is_q8:
            fallback='\nint dotPacked4x8EXT(int a,int b){int s=0;for(int i=0;i<4;++i)s+=bitfieldExtract(a,i*8,8)*bitfieldExtract(b,i*8,8);return s;}\n'
            text=text.replace('void main()',fallback+'\nvoid main()')
        shader=self.fn('glCreateShader',U,[U])(0x91B9);bs=text.encode();ptr=C.c_char_p(bs);length=I(len(bs))
        self.fn('glShaderSource',None,[U,I,C.POINTER(C.c_char_p),C.POINTER(I)])(shader,1,C.byref(ptr),C.byref(length))
        self.fn('glCompileShader',None,[U])(shader);status=I();self.fn('glGetShaderiv',None,[U,U,C.POINTER(I)])(shader,0x8B81,C.byref(status))
        if not status.value:
            log=C.create_string_buffer(32768);n=I();self.fn('glGetShaderInfoLog',None,[U,I,C.POINTER(I),V])(shader,len(log),C.byref(n),log);raise RuntimeError(name+' shader compile:\n'+log.value.decode())
        program=self.fn('glCreateProgram',U,[])();self.fn('glAttachShader',None,[U,U])(program,shader);self.fn('glLinkProgram',None,[U])(program)
        self.fn('glGetProgramiv',None,[U,U,C.POINTER(I)])(program,0x8B82,C.byref(status))
        if not status.value:
            log=C.create_string_buffer(32768);n=I();self.fn('glGetProgramInfoLog',None,[U,I,C.POINTER(I),V])(program,len(log),C.byref(n),log);raise RuntimeError(name+' link:\n'+log.value.decode())
        self.fn('glDeleteShader',None,[U])(shader);self.programs[name]=program;return program
    def run(self,fixture,shaders):
        lines=(fixture/'run.plan').read_text().splitlines();dumps=[]
        for line in lines:
            words=shlex.split(line)
            if words and words[0]=='dump':
                (fixture/words[2]).unlink(missing_ok=True)
        (fixture/'execution.json').unlink(missing_ok=True)
        if lines[0]!='MAPLE_TQ2_PLAN 1':raise ValueError('bad plan')
        for line in lines[1:]:
            w=shlex.split(line)
            if not w:continue
            if w[0]=='require_dot':continue
            if w[0]=='buffer':
                _,name,size,file=w;size=int(size);data=b'\0'*size if file=='-' else (fixture/file).read_bytes()
                if len(data)!=size:raise ValueError('buffer size mismatch')
                self.buffers[name]=(self.new_buffer(0x90D2,data),size)
            elif w[0]=='fill':
                _,name,offset,size,value=w;b,_=self.buffers[name];val=U(int(value));self.fn('glBindBuffer',None,[U,U])(0x90D2,b)
                self.fn('glClearBufferSubData',None,[U,U,S,S,U,U,V])(0x90D2,0x8236,int(offset),int(size),0x8D94,0x1405,C.byref(val))
            elif w[0]=='dispatch':
                name=w[1];grid=list(map(int,w[2:5]));n=int(w[5]);push=list(map(int,w[6:6+n]));nb=int(w[6+n]);bindings=w[7+n:7+n+nb]
                prog=self.program(shaders,name);self.fn('glUseProgram',None,[U])(prog)
                for i in range(9):self.fn('glBindBufferBase',None,[U,U,U])(0x90D2,i,self.buffers[bindings[i] if i<nb else 'dummy'][0])
                data=struct.pack('<32I',*(push+[0]*(32-len(push))));arr=C.create_string_buffer(data)
                self.fn('glBindBuffer',None,[U,U])(0x8A11,self.uniform);self.fn('glBufferSubData',None,[U,S,S,V])(0x8A11,0,128,arr)
                self.fn('glBindBufferBase',None,[U,U,U])(0x8A11,0,self.uniform)
                self.fn('glDispatchCompute',None,[U,U,U])(*grid)
                print('EXEC',name,grid,flush=True)
            elif w[0]=='dump':dumps.append(w[1:])
            else:raise ValueError('unknown plan op '+w[0])
            self.fn('glMemoryBarrier',None,[U])(0xffffffff);self.err(w[0])
        self.fn('glFinish',None,[])()
        for name,file in dumps:
            b,size=self.buffers[name];data=(C.c_ubyte*size)();self.fn('glBindBuffer',None,[U,U])(0x90D2,b)
            self.fn('glGetBufferSubData',None,[U,S,S,V])(0x90D2,0,size,data);(fixture/file).write_bytes(bytes(data))
        receipt={'backend':'Mesa EGL/OpenGL software oracle','renderer':self.renderer,'version':self.version,'native_integer_dot_tested':False,'vulkan_tested':False,'time':time.time()}
        (fixture/'execution.json').write_text(json.dumps(receipt,indent=2))
    def close(self):
        for p in self.programs.values():self.fn('glDeleteProgram',None,[U])(p)
        ids=[b for b,_ in self.buffers.values()]+[self.uniform]
        if ids:self.fn('glDeleteBuffers',None,[I,C.POINTER(U)])(len(ids),(U*len(ids))(*ids))
        self.egl.eglMakeCurrent(self.display,None,None,None)
        self.egl.eglDestroyContext.argtypes=[V,V];self.egl.eglDestroyContext(self.display,self.context)
        self.egl.eglTerminate.argtypes=[V];self.egl.eglTerminate(self.display)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('fixture',type=Path);a=ap.parse_args();o=Oracle()
    try:o.run(a.fixture,Path(__file__).resolve().parents[1]/'shaders')
    finally:o.close()
if __name__=='__main__':main()
