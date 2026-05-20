#include "ggml-backend.h"
#include "llama-moe-offloader.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" void *
llama_moe_offloader_create_shared_event(ggml_backend_t backend) {
    (void)backend;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLSharedEvent> ev = [dev newSharedEvent];
    [ev retain];
    return (__bridge void *)ev;
}

extern "C" void llama_moe_offloader_signal_event(void *event, uint64_t value) {
    id<MTLSharedEvent> ev = (__bridge id<MTLSharedEvent>)event;
    ev.signaledValue = value;
}

extern "C" void llama_moe_offloader_release_event(void *event) {
    id<MTLSharedEvent> ev = (__bridge id<MTLSharedEvent>)event;
    [ev release];
}
