#import "ggml-ane-runtime.h"

#import <Foundation/Foundation.h>
#include <unistd.h>
#import <objc/message.h>
#import <dlfcn.h>

#if !__has_feature(objc_arc)
#error "ggml-ane-runtime.m must be compiled with -fobjc-arc"
#endif

#define ANE_QOS 21

static Class g_desc_cls;
static Class g_model_cls;
static Class g_request_cls;
static Class g_iosurface_cls;

static bool ane_load(void) {
    static bool loaded = false;
    static bool ok     = false;
    if (loaded) {
        return ok;
    }
    loaded = true;
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        return false;
    }
    g_desc_cls      = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_model_cls     = NSClassFromString(@"_ANEInMemoryModel");
    g_request_cls   = NSClassFromString(@"_ANERequest");
    g_iosurface_cls = NSClassFromString(@"_ANEIOSurfaceObject");
    ok = g_desc_cls && g_model_cls && g_request_cls && g_iosurface_cls;
    return ok;
}

bool ggml_ane_runtime_available(void) {
    return ane_load();
}

const char * ggml_ane_runtime_arch(void) {
    static char arch[32] = "unknown";
    static bool done = false;
    if (!done && ane_load()) {
        done = true;
        Class info = NSClassFromString(@"_ANEDeviceInfo");
        if (info && [info respondsToSelector:@selector(aneSubType)]) {
            NSString * s = ((id (*)(Class, SEL))objc_msgSend)(info, @selector(aneSubType));
            if ([s isKindOfClass:[NSString class]]) {
                snprintf(arch, sizeof(arch), "%s", s.UTF8String);
            }
        }
    }
    return arch;
}

IOSurfaceRef ggml_ane_surface_create(size_t size) {
    return IOSurfaceCreate((__bridge CFDictionaryRef) @{
        (id) kIOSurfaceWidth:           @(size),
        (id) kIOSurfaceHeight:          @1,
        (id) kIOSurfaceBytesPerElement: @1,
        (id) kIOSurfaceBytesPerRow:     @(size),
        (id) kIOSurfaceAllocSize:       @(size),
        (id) kIOSurfacePixelFormat:     @0,
    });
}

// surface over existing memory (page aligned): the ANE reads/writes that memory directly
IOSurfaceRef ggml_ane_surface_wrap(void * ptr, size_t size) {
    if (((uintptr_t) ptr) % (uintptr_t) sysconf(_SC_PAGESIZE) != 0) {
        return NULL;
    }
    return IOSurfaceCreate((__bridge CFDictionaryRef) @{
        (id) kIOSurfaceWidth:           @(size),
        (id) kIOSurfaceHeight:          @1,
        (id) kIOSurfaceBytesPerElement: @1,
        (id) kIOSurfaceBytesPerRow:     @(size),
        (id) kIOSurfaceAllocSize:       @(size),
        (id) kIOSurfacePixelFormat:     @0,
        @"IOSurfaceAddress":            @((uint64_t) (uintptr_t) ptr),
    });
}

// one function of the program (a program can hold several functions, e.g. one per input shape, sharing its weights)
typedef struct {
    char               name[GGML_ANE_MAX_NAME];
    int                n_in;
    int                n_out;
    ggml_ane_io_desc * in;
    ggml_ane_io_desc * out;
    int              * in_sym;   // global symbol index of each input/output (used in requests)
    int              * out_sym;
    CFTypeRef          request;  // retained _ANERequest for the last bound surfaces
    IOSurfaceRef     * bound;    // n_in + n_out surfaces used by request
} ggml_ane_proc;

struct ggml_ane_program {
    CFTypeRef       model;       // retained _ANEInMemoryModel
    int             n_proc;
    ggml_ane_proc * procs;       // by procedure index
};

static void set_err(char * err, size_t err_len, NSError * e, const char * what) {
    if (!err || err_len == 0) {
        return;
    }
    NSError * u = e.userInfo[NSUnderlyingErrorKey];
    NSString * d = (u ? u : e).localizedDescription;
    snprintf(err, err_len, "%s: %s", what, d ? d.UTF8String : "unknown error");
}

static void io_from_dict(ggml_ane_io_desc * io, NSDictionary * d) {
    NSString * name = d[@"Name"];
    if ([name hasSuffix:@"@output"]) {
        name = [name substringToIndex:name.length - 7];
    }
    snprintf(io->name, sizeof(io->name), "%s", name.UTF8String);
    io->n            = [d[@"Batches"]     longLongValue];
    io->c            = [d[@"Channels"]    longLongValue];
    io->h            = [d[@"Height"]      longLongValue];
    io->w            = [d[@"Width"]       longLongValue];
    io->batch_stride = [d[@"BatchStride"] unsignedLongValue];
    io->plane_stride = [d[@"PlaneStride"] unsignedLongValue];
    io->row_stride   = [d[@"RowStride"]   unsignedLongValue];
    io->esize        = [d[@"Type"] isEqual:@"Float32"] ? 4 : 2;
    if (io->n < 1) {
        io->n = 1;
    }
}

// after a compile fails for lack of storage, refuse further compiles in this process: failed compiles can leave files
// in the OS cache (on iOS they are not reclaimable by the app) and the next ones would fail the same way
static bool g_compile_failed = false;

ggml_ane_program * ggml_ane_program_create(const char * mil, size_t mil_len, const void * weights, size_t weights_len, char * err, size_t err_len) {
    if (g_compile_failed) {
        snprintf(err, err_len, "an earlier compile failed for lack of storage: refusing to compile");
        return NULL;
    }
    if (!ane_load()) {
        snprintf(err, err_len, "AppleNeuralEngine framework not available");
        return NULL;
    }
    @autoreleasepool {
        NSData * mil_data = [NSData dataWithBytes:mil length:mil_len];
        // the descriptor keeps the weights for the lifetime of the program: hand it a file mapping (clean pages, not
        // counted in the app footprint) instead of a copy in memory; the file is unlinked right away
        NSData * w_data = nil;
        if (weights_len) {
            NSString * path = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"ggml-ane-w-%@.bin", NSUUID.UUID.UUIDString]];
            if ([[NSData dataWithBytesNoCopy:(void *) (uintptr_t) weights length:weights_len freeWhenDone:NO] writeToFile:path atomically:NO]) {
                w_data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedAlways error:nil];
                unlink(path.fileSystemRepresentation);
            }
        }
        if (weights_len && !w_data) {
            w_data = [NSData dataWithBytes:weights length:weights_len];
        }
        NSDictionary * wdict = w_data ? @{ @"@model_path/weights/weight.bin": @{ @"offset": @0, @"data": w_data } } : @{};

        id desc = ((id (*)(Class, SEL, id, id, id)) objc_msgSend)(g_desc_cls, @selector(modelWithMILText:weights:optionsPlist:), mil_data, wdict, nil);
        if (!desc) {
            snprintf(err, err_len, "invalid model descriptor");
            return NULL;
        }
        id model = ((id (*)(Class, SEL, id)) objc_msgSend)(g_model_cls, @selector(inMemoryModelWithDescriptor:), desc);
        NSError * e = nil;

        // the compiler reads model.mil and weights from <tmp>/<hex id>/; the result is cached by the OS
        // keep the directory until after load: load sometimes needs it (sandbox extension for the path)
        NSString * dir = nil;
        NSFileManager * fm = [NSFileManager defaultManager];
        BOOL cached = ((BOOL (*)(id, SEL)) objc_msgSend)(model, @selector(compiledModelExists));
        if (!cached) {
            NSString * hex = ((id (*)(id, SEL)) objc_msgSend)(model, @selector(hexStringIdentifier));
            dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
            [fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
            [mil_data writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:NO];
            if (w_data) {
                [w_data writeToFile:[dir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:NO];
            }
            if (!((BOOL (*)(id, SEL, unsigned, id, NSError **)) objc_msgSend)(model, @selector(compileWithQoS:options:error:), ANE_QOS, @{}, &e)) {
                set_err(err, err_len, e, "compile failed");
                const NSString * desc = e.description ?: @"";
                if ([desc containsString:@"file backing"] || [desc containsString:@"No space"]) {
                    g_compile_failed = true;
                }
                [fm removeItemAtPath:dir error:nil];
                return NULL;
            }
        }
        BOOL loaded = ((BOOL (*)(id, SEL, unsigned, id, NSError **)) objc_msgSend)(model, @selector(loadWithQoS:options:error:), ANE_QOS, @{}, &e);
        if (dir) {
            [fm removeItemAtPath:dir error:nil];
        }
        if (!loaded) {
            set_err(err, err_len, e, "load failed");
            return NULL;
        }

        // procedures: ids from the name map, symbols from the per-procedure index arrays (global symbol indices),
        // layouts from the NetworkStatusList entry with the same name
        NSDictionary * attrs = ((id (*)(id, SEL)) objc_msgSend)(model, @selector(modelAttributes));
        NSDictionary * mdesc = attrs[@"ANEFModelDescription"];
        NSArray      * in_syms  = mdesc[@"kANEFModelInputSymbolsArrayKey"];
        NSArray      * out_syms = mdesc[@"kANEFModelOutputSymbolsArrayKey"];
        NSArray      * procs    = mdesc[@"ANEFModelProcedures"];
        NSDictionary * name_map = mdesc[@"kANEFModelProcedureNameToIDMapKey"];
        NSArray      * nsl      = attrs[@"NetworkStatusList"];

        ggml_ane_program * prog = calloc(1, sizeof(*prog));
        prog->n_proc = procs.count > 0 ? (int) procs.count : 1;
        prog->procs  = calloc(prog->n_proc, sizeof(ggml_ane_proc));
        for (int p = 0; p < prog->n_proc; p++) {
            ggml_ane_proc * pr = &prog->procs[p];
            NSString * name = @"main";
            for (NSString * k in name_map) {
                if ([name_map[k] intValue] == p) {
                    name = k;
                }
            }
            snprintf(pr->name, sizeof(pr->name), "%s", name.UTF8String);
            NSDictionary * st = nsl.count > (NSUInteger) p ? nsl[p] : nsl[0];
            for (NSDictionary * d in nsl) {
                if ([d[@"Name"] isEqual:name]) {
                    st = d;
                }
            }
            NSMutableDictionary * by_name = [NSMutableDictionary dictionary];
            for (NSDictionary * d in st[@"LiveInputList"])  by_name[d[@"Name"]] = d;
            for (NSDictionary * d in st[@"LiveOutputList"]) by_name[d[@"Name"]] = d;
            NSArray * isx = procs.count ? procs[p][@"ANEFModelInputSymbolIndexArray"]  : nil;
            NSArray * osx = procs.count ? procs[p][@"ANEFModelOutputSymbolIndexArray"] : nil;
            pr->n_in  = isx.count ? (int) isx.count : (int) in_syms.count;
            pr->n_out = osx.count ? (int) osx.count : (int) out_syms.count;
            pr->in      = calloc(pr->n_in  + 1, sizeof(ggml_ane_io_desc));
            pr->out     = calloc(pr->n_out + 1, sizeof(ggml_ane_io_desc));
            pr->in_sym  = calloc(pr->n_in  + 1, sizeof(int));
            pr->out_sym = calloc(pr->n_out + 1, sizeof(int));
            pr->bound   = calloc(pr->n_in + pr->n_out + 1, sizeof(IOSurfaceRef));
            for (int i = 0; i < pr->n_in; i++) {
                pr->in_sym[i] = isx.count ? [isx[i] intValue] : i;
                io_from_dict(&pr->in[i], by_name[in_syms[pr->in_sym[i]]]);
            }
            for (int i = 0; i < pr->n_out; i++) {
                pr->out_sym[i] = osx.count ? [osx[i] intValue] : i;
                io_from_dict(&pr->out[i], by_name[out_syms[pr->out_sym[i]]]);
            }
        }
        prog->model = CFBridgingRetain(model);
        return prog;
    }
}

void ggml_ane_program_free(ggml_ane_program * prog) {
    if (!prog) {
        return;
    }
    @autoreleasepool {
        id model = (__bridge id) prog->model;
        NSError * e = nil;
        ((BOOL (*)(id, SEL, unsigned, NSError **)) objc_msgSend)(model, @selector(unloadWithQoS:error:), ANE_QOS, &e);
        // debug: the OS keeps every compiled program on disk; GGML_ANE_PURGE_CACHE drops it when the program is freed
        const bool purge = getenv("GGML_ANE_PURGE_CACHE") != NULL;
        if (purge && [model respondsToSelector:@selector(purgeCompiledModel)]) {
            ((void (*)(id, SEL)) objc_msgSend)(model, @selector(purgeCompiledModel));
        }
        CFRelease(prog->model);
        for (int p = 0; p < prog->n_proc; p++) {
            if (prog->procs[p].request) {
                CFRelease(prog->procs[p].request);
            }
        }
    }
    for (int p = 0; p < prog->n_proc; p++) {
        free(prog->procs[p].in);
        free(prog->procs[p].out);
        free(prog->procs[p].in_sym);
        free(prog->procs[p].out_sym);
        free(prog->procs[p].bound);
    }
    free(prog->procs);
    free(prog);
}

void ggml_ane_program_purge_compiled(ggml_ane_program * prog) {
    @autoreleasepool {
        id model = (__bridge id) prog->model;
        if ([model respondsToSelector:@selector(purgeCompiledModel)]) {
            ((void (*)(id, SEL)) objc_msgSend)(model, @selector(purgeCompiledModel));
        }
    }
}

int ggml_ane_program_n_procedures(const ggml_ane_program * prog) {
    return prog->n_proc;
}

int ggml_ane_program_procedure(const ggml_ane_program * prog, const char * name) {
    for (int p = 0; p < prog->n_proc; p++) {
        if (strcmp(prog->procs[p].name, name) == 0) {
            return p;
        }
    }
    return -1;
}

int ggml_ane_program_n_inputs(const ggml_ane_program * prog, int proc) {
    return prog->procs[proc].n_in;
}

int ggml_ane_program_n_outputs(const ggml_ane_program * prog, int proc) {
    return prog->procs[proc].n_out;
}

const ggml_ane_io_desc * ggml_ane_program_input(const ggml_ane_program * prog, int proc, int i) {
    return &prog->procs[proc].in[i];
}

const ggml_ane_io_desc * ggml_ane_program_output(const ggml_ane_program * prog, int proc, int i) {
    return &prog->procs[proc].out[i];
}

bool ggml_ane_program_eval(ggml_ane_program * prog, int proc, IOSurfaceRef * inputs, IOSurfaceRef * outputs, char * err, size_t err_len) {
    @autoreleasepool {
        ggml_ane_proc * pr = &prog->procs[proc];
        // rebuild the request only when the bound surfaces change
        bool same = pr->request != NULL;
        for (int i = 0; same && i < pr->n_in;  i++) same = pr->bound[i] == inputs[i];
        for (int i = 0; same && i < pr->n_out; i++) same = pr->bound[pr->n_in + i] == outputs[i];
        if (!same) {
            NSMutableArray * in_objs  = [NSMutableArray array];
            NSMutableArray * in_idx   = [NSMutableArray array];
            NSMutableArray * out_objs = [NSMutableArray array];
            NSMutableArray * out_idx  = [NSMutableArray array];
            for (int i = 0; i < pr->n_in; i++) {
                [in_objs addObject:((id (*)(Class, SEL, IOSurfaceRef)) objc_msgSend)(g_iosurface_cls, @selector(objectWithIOSurface:), inputs[i])];
                [in_idx addObject:@(pr->in_sym[i])];
                pr->bound[i] = inputs[i];
            }
            for (int i = 0; i < pr->n_out; i++) {
                [out_objs addObject:((id (*)(Class, SEL, IOSurfaceRef)) objc_msgSend)(g_iosurface_cls, @selector(objectWithIOSurface:), outputs[i])];
                [out_idx addObject:@(pr->out_sym[i])];
                pr->bound[pr->n_in + i] = outputs[i];
            }
            id req = ((id (*)(Class, SEL, id, id, id, id, id, id, id)) objc_msgSend)(g_request_cls,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                in_objs, in_idx, out_objs, out_idx, nil, nil, @(proc));
            if (pr->request) {
                CFRelease(pr->request);
            }
            pr->request = CFBridgingRetain(req);
        }
        NSError * e = nil;
        BOOL ok = ((BOOL (*)(id, SEL, unsigned, id, id, NSError **)) objc_msgSend)((__bridge id) prog->model,
            @selector(evaluateWithQoS:options:request:error:), ANE_QOS, @{}, (__bridge id) pr->request, &e);
        if (!ok) {
            set_err(err, err_len, e, "evaluate failed");
        }
        return ok;
    }
}
