# Capstone Project: Mobile LLM Application

**Difficulty**: Intermediate-Advanced
**Estimated Time**: 35-50 hours
**Modules Required**: 1, 3, 8
**Prerequisites**: Swift/Kotlin, Mobile Development, Metal/Vulkan

---

## Project Overview

Build a native mobile app (iOS/Android) running LLM inference on-device using llama.cpp.

**Platforms**: 
- iOS (Swift, Metal acceleration)
- Android (Kotlin, Vulkan/OpenCL)

**Features**:
- On-device inference (privacy-first)
- Quantized 3B model (Q4_K_M)
- Streaming responses
- Conversation management
- Model download & caching

---

## Implementation

### iOS (Swift + Metal)

**Phase 1**: llama.cpp Integration
- Compile llama.cpp for iOS (arm64)
- Metal backend enabled
- Swift bindings (C interop)

**Phase 2**: UI Development
- SwiftUI chat interface
- Message bubbles with markdown
- Token-by-token streaming
- Settings panel

**Phase 3**: Optimization
- Background processing
- Battery optimization
- Memory management
- Model compression

**Example Code**:
```swift
import Foundation

class LlamaModel {
    private var context: OpaquePointer?
    
    func load(modelPath: String) {
        let params = llama_context_default_params()
        context = llama_load_model_from_file(modelPath, params)
    }
    
    func generate(prompt: String) async -> AsyncStream<String> {
        AsyncStream { continuation in
            // Token-by-token generation
            // Update UI with continuation.yield(token)
        }
    }
}
```

### Android (Kotlin + Vulkan)

**Phase 1**: JNI Integration
- Build llama.cpp with Vulkan
- JNI bindings
- NDK configuration

**Phase 2**: Jetpack Compose UI
- Material Design 3
- Chat interface
- File picker for models

**Phase 3**: Optimization
- WorkManager for background tasks
- Room for conversation storage
- Vulkan acceleration

---

## Challenges & Solutions

**Challenge 1**: Model Size
- Solution: Quantization (Q3_K_M for <2GB), on-demand download

**Challenge 2**: Battery Life
- Solution: Inference throttling, batch scheduling

**Challenge 3**: Memory Constraints
- Solution: Small context (1K), aggressive KV cache limits

---

## Evaluation

- **Performance**: 5+ tokens/sec on flagship phones
- **User Experience**: Smooth UI, <3s first token
- **Resource Usage**: <2GB RAM, reasonable battery drain
- **Code Quality**: Clean architecture, testable

---

**Deliverables**: iOS & Android apps, App Store screenshots, Demo video
