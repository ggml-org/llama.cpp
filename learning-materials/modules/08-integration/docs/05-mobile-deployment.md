# Mobile Deployment - Android and iOS

**Module 8, Lesson 5**
**Estimated Time**: 3-4 hours
**Difficulty**: Advanced

## Overview

Deploy llama.cpp on mobile devices for on-device AI inference. Learn to build Android and iOS applications with local LLM capabilities, optimize for mobile constraints, and create responsive user experiences.

## Learning Objectives

- Build Android apps with llama.cpp
- Create iOS apps with llama.cpp
- Optimize models for mobile devices
- Handle mobile-specific constraints
- Implement efficient mobile UIs

## Prerequisites

- Android Studio or Xcode
- Basic Android/iOS development knowledge
- Understanding of mobile app architecture
- Module 8, Lessons 1-3

---

## 1. Mobile Deployment Challenges

### Key Constraints

| Constraint | Desktop | Mobile | Impact |
|------------|---------|--------|--------|
| RAM | 16-64 GB | 2-8 GB | Model size limits |
| Storage | TBs | 64-512 GB | Model storage |
| CPU | High-end | ARM | Slower inference |
| Battery | AC power | Limited | Power efficiency critical |
| Thermal | Active cooling | Passive | Throttling issues |
| Network | Fast, stable | Variable | Offline capability needed |

### Optimization Strategies

1. **Model Quantization**: Use Q4_K_M or smaller
2. **Context Limits**: Reduce n_ctx to 512-2048
3. **Batching**: Single-sequence inference
4. **Streaming**: Progressive UI updates
5. **Background Processing**: Async inference
6. **Caching**: Cache embeddings and KV cache

---

## 2. Android Integration

### Setup with Android NDK

#### Build Configuration (build.gradle)

```gradle
// app/build.gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'com.example.llamacpp'
    compileSdk 34

    defaultConfig {
        applicationId "com.example.llamacpp"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"

        // NDK Configuration
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }

        externalNativeBuild {
            cmake {
                cppFlags '-std=c++17'
                arguments '-DGGML_CUDA=OFF',
                          '-DLLAMA_BUILD_TESTS=OFF',
                          '-DLLAMA_BUILD_EXAMPLES=OFF'
            }
        }
    }

    externalNativeBuild {
        cmake {
            path file('src/main/cpp/CMakeLists.txt')
            version '3.22.1'
        }
    }

    buildFeatures {
        viewBinding true
        compose true
    }
}

dependencies {
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.11.0'

    // Jetpack Compose
    implementation platform('androidx.compose:compose-bom:2024.01.00')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.material3:material3'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.7.0'
    implementation 'androidx.activity:activity-compose:1.8.2'

    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

#### CMakeLists.txt

```cmake
# src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22.1)
project(llama-android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add llama.cpp sources
add_subdirectory(llama.cpp)

# JNI wrapper library
add_library(
    llama-jni
    SHARED
    llama_jni.cpp
)

target_link_libraries(
    llama-jni
    llama
    ggml
    log
)
```

### JNI Wrapper

```cpp
// src/main/cpp/llama_jni.cpp
#include <jni.h>
#include <string>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "LlamaCPP"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_llamacpp_LlamaModel_nativeInit(
    JNIEnv* env,
    jobject /* this */,
    jstring model_path,
    jint n_ctx,
    jint n_threads
) {
    const char* path = env->GetStringUTFChars(model_path, nullptr);

    // Initialize model parameters
    llama_model_params model_params = llama_model_default_params();

    // Load model
    llama_model* model = llama_load_model_from_file(path, model_params);

    env->ReleaseStringUTFChars(model_path, path);

    if (!model) {
        LOGI("Failed to load model");
        return 0;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    if (!ctx) {
        llama_free_model(model);
        return 0;
    }

    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_com_example_llamacpp_LlamaModel_nativeGenerate(
    JNIEnv* env,
    jobject /* this */,
    jlong ctx_ptr,
    jstring prompt,
    jint max_tokens,
    jobject callback
) {
    llama_context* ctx = reinterpret_cast<llama_context*>(ctx_ptr);
    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);

    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(strlen(prompt_str) + 1);
    int n_tokens = llama_tokenize(
        ctx,
        prompt_str,
        tokens.data(),
        tokens.size(),
        true
    );
    tokens.resize(n_tokens);

    env->ReleaseStringUTFChars(prompt, prompt_str);

    // Evaluate prompt
    llama_eval(ctx, tokens.data(), tokens.size(), 0);

    // Get callback method
    jclass callback_class = env->GetObjectClass(callback);
    jmethodID on_token = env->GetMethodID(
        callback_class,
        "onToken",
        "(Ljava/lang/String;)V"
    );

    // Generate tokens
    for (int i = 0; i < max_tokens; i++) {
        // Sample next token
        llama_token token = llama_sample_top_p_top_k(
            ctx,
            nullptr,
            0,
            40,    // top_k
            0.95f, // top_p
            0.8f,  // temp
            1.0f   // repeat_penalty
        );

        if (token == llama_token_eos(ctx)) {
            break;
        }

        // Convert to string
        char buf[128];
        llama_token_to_str(ctx, token, buf, sizeof(buf));

        // Call Java callback
        jstring token_str = env->NewStringUTF(buf);
        env->CallVoidMethod(callback, on_token, token_str);
        env->DeleteLocalRef(token_str);

        // Evaluate token
        llama_eval(ctx, &token, 1, tokens.size() + i);
    }
}

JNIEXPORT void JNICALL
Java_com_example_llamacpp_LlamaModel_nativeFree(
    JNIEnv* env,
    jobject /* this */,
    jlong ctx_ptr
) {
    llama_context* ctx = reinterpret_cast<llama_context*>(ctx_ptr);
    if (ctx) {
        llama_model* model = llama_get_model(ctx);
        llama_free(ctx);
        llama_free_model(model);
    }
}

} // extern "C"
```

### Kotlin Wrapper

```kotlin
// LlamaModel.kt
package com.example.llamacpp

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import java.io.File

class LlamaModel(
    private val context: Context
) {
    private var nativeHandle: Long = 0

    companion object {
        init {
            System.loadLibrary("llama-jni")
        }
    }

    fun load(
        modelPath: String,
        nCtx: Int = 2048,
        nThreads: Int = 4
    ): Boolean {
        nativeHandle = nativeInit(modelPath, nCtx, nThreads)
        return nativeHandle != 0L
    }

    suspend fun generate(
        prompt: String,
        maxTokens: Int = 512
    ): Flow<String> = flow {
        withContext(Dispatchers.IO) {
            nativeGenerate(
                nativeHandle,
                prompt,
                maxTokens,
                object : GenerationCallback {
                    override fun onToken(token: String) {
                        // Emit token to Flow
                        kotlinx.coroutines.runBlocking {
                            emit(token)
                        }
                    }
                }
            )
        }
    }

    fun free() {
        if (nativeHandle != 0L) {
            nativeFree(nativeHandle)
            nativeHandle = 0
        }
    }

    // Native methods
    private external fun nativeInit(
        modelPath: String,
        nCtx: Int,
        nThreads: Int
    ): Long

    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        callback: GenerationCallback
    )

    private external fun nativeFree(handle: Long)

    interface GenerationCallback {
        fun onToken(token: String)
    }
}
```

### Android UI (Jetpack Compose)

```kotlin
// MainActivity.kt
package com.example.llamacpp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private lateinit var llama: LlamaModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize model
        llama = LlamaModel(this)

        lifecycleScope.launch {
            val modelPath = "${filesDir.absolutePath}/model.gguf"
            llama.load(modelPath)
        }

        setContent {
            MaterialTheme {
                ChatScreen(llama)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        llama.free()
    }
}

@Composable
fun ChatScreen(llama: LlamaModel) {
    var messages by remember { mutableStateOf(listOf<Message>()) }
    var input by remember { mutableStateOf("") }
    var isGenerating by remember { mutableStateOf(false) }

    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Messages
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(16.dp)
        ) {
            items(messages) { message ->
                MessageBubble(message)
                Spacer(modifier = Modifier.height(8.dp))
            }
        }

        // Input
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Type a message...") },
                enabled = !isGenerating
            )

            Spacer(modifier = Modifier.width(8.dp))

            Button(
                onClick = {
                    scope.launch {
                        isGenerating = true

                        // Add user message
                        messages = messages + Message(input, true)
                        val prompt = input
                        input = ""

                        // Generate response
                        var response = ""
                        messages = messages + Message("", false)

                        llama.generate(prompt).collect { token ->
                            response += token
                            messages = messages.dropLast(1) + Message(response, false)
                        }

                        isGenerating = false
                    }
                },
                enabled = input.isNotBlank() && !isGenerating
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
fun MessageBubble(message: Message) {
    Surface(
        color = if (message.isUser) {
            MaterialTheme.colorScheme.primary
        } else {
            MaterialTheme.colorScheme.secondaryContainer
        },
        shape = MaterialTheme.shapes.medium
    ) {
        Text(
            text = message.text,
            modifier = Modifier.padding(12.dp),
            color = if (message.isUser) {
                MaterialTheme.colorScheme.onPrimary
            } else {
                MaterialTheme.colorScheme.onSecondaryContainer
            }
        )
    }
}

data class Message(
    val text: String,
    val isUser: Boolean
)
```

---

## 3. iOS Integration

### Swift Wrapper

```swift
// LlamaModel.swift
import Foundation

class LlamaModel {
    private var context: OpaquePointer?
    private var model: OpaquePointer?

    func load(modelPath: String, nCtx: Int32 = 2048) -> Bool {
        // Model parameters
        var modelParams = llama_model_default_params()

        // Load model
        model = llama_load_model_from_file(modelPath, modelParams)
        guard model != nil else {
            print("Failed to load model")
            return false
        }

        // Context parameters
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(nCtx)
        ctxParams.n_threads = UInt32(ProcessInfo.processInfo.activeProcessorCount)

        // Create context
        context = llama_new_context_with_model(model, ctxParams)
        guard context != nil else {
            llama_free_model(model)
            return false
        }

        return true
    }

    func generate(
        prompt: String,
        maxTokens: Int = 512,
        onToken: @escaping (String) -> Void
    ) {
        guard let context = context else { return }

        // Tokenize prompt
        var tokens = [llama_token](repeating: 0, count: prompt.count + 1)
        let nTokens = llama_tokenize(
            context,
            prompt,
            Int32(prompt.count),
            &tokens,
            Int32(tokens.count),
            true
        )
        tokens = Array(tokens.prefix(Int(nTokens)))

        // Evaluate prompt
        llama_eval(context, tokens, Int32(tokens.count), 0, 0)

        // Generate tokens
        for i in 0..<maxTokens {
            // Sample next token
            let token = llama_sample_top_p_top_k(
                context,
                nil,
                0,
                40,    // top_k
                0.95,  // top_p
                0.8,   // temp
                1.0    // repeat_penalty
            )

            if token == llama_token_eos(context) {
                break
            }

            // Convert to string
            var buffer = [CChar](repeating: 0, count: 128)
            llama_token_to_str(context, token, &buffer, buffer.count)
            let tokenStr = String(cString: buffer)

            onToken(tokenStr)

            // Evaluate token
            llama_eval(context, [token], 1, Int32(tokens.count + i), 0)
        }
    }

    deinit {
        if let context = context {
            llama_free(context)
        }
        if let model = model {
            llama_free_model(model)
        }
    }
}
```

### SwiftUI Chat Interface

```swift
// ContentView.swift
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var inputText = ""

    var body: some View {
        VStack {
            // Messages
            ScrollView {
                ScrollViewReader { proxy in
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageView(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                    .onChange(of: viewModel.messages.count) { _ in
                        if let lastMessage = viewModel.messages.last {
                            withAnimation {
                                proxy.scrollTo(lastMessage.id, anchor: .bottom)
                            }
                        }
                    }
                }
            }

            // Input
            HStack {
                TextField("Type a message...", text: $inputText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .disabled(viewModel.isGenerating)

                Button(action: {
                    viewModel.sendMessage(inputText)
                    inputText = ""
                }) {
                    Image(systemName: "paperplane.fill")
                }
                .disabled(inputText.isEmpty || viewModel.isGenerating)
            }
            .padding()
        }
        .navigationTitle("LLaMA Chat")
    }
}

struct MessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }

            Text(message.text)
                .padding(12)
                .background(message.isUser ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(message.isUser ? .white : .primary)
                .cornerRadius(12)

            if !message.isUser {
                Spacer()
            }
        }
    }
}

class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var isGenerating = false

    private let llama = LlamaModel()

    init() {
        // Load model
        let modelPath = Bundle.main.path(forResource: "model", ofType: "gguf")!
        llama.load(modelPath: modelPath)
    }

    func sendMessage(_ text: String) {
        // Add user message
        messages.append(ChatMessage(text: text, isUser: true))

        isGenerating = true
        var response = ""

        // Generate response
        DispatchQueue.global(qos: .userInitiated).async {
            self.llama.generate(prompt: text) { token in
                DispatchQueue.main.async {
                    response += token

                    // Update or create assistant message
                    if let lastMessage = self.messages.last,
                       !lastMessage.isUser {
                        self.messages[self.messages.count - 1] = ChatMessage(
                            text: response,
                            isUser: false
                        )
                    } else {
                        self.messages.append(ChatMessage(
                            text: response,
                            isUser: false
                        ))
                    }
                }
            }

            DispatchQueue.main.async {
                self.isGenerating = false
            }
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
}
```

---

## 4. Model Optimization for Mobile

### Recommended Quantizations

```
Model Size | Quantization | RAM Required | Quality
-----------|--------------|--------------|--------
  2-3 GB   | Q4_K_M      | 4 GB         | Good
  1-2 GB   | Q4_0        | 3 GB         | Acceptable
  <1 GB    | Q3_K_S      | 2 GB         | Lower quality
```

### Model Selection

```python
# Recommended models for mobile
MOBILE_MODELS = {
    "flagship": {  # High-end phones (8GB+ RAM)
        "model": "llama-2-7b-chat.Q4_K_M.gguf",
        "n_ctx": 2048,
        "size_gb": 3.8
    },
    "mid_range": {  # Mid-range phones (6GB RAM)
        "model": "phi-2.Q4_0.gguf",
        "n_ctx": 1024,
        "size_gb": 1.6
    },
    "budget": {  # Budget phones (4GB RAM)
        "model": "tinyllama-1.1b.Q4_0.gguf",
        "n_ctx": 512,
        "size_gb": 0.6
    }
}
```

---

## Summary

In this lesson, you learned:
- ✅ Building Android apps with llama.cpp
- ✅ Creating iOS apps with Swift
- ✅ Mobile optimization strategies
- ✅ Handling resource constraints
- ✅ Building mobile chat UIs

## Next Steps

- **Lesson 6**: Web Integration
- **Lab 8.5**: Build a mobile app
- **Project**: Cross-platform mobile chat

## Additional Resources

- [Android NDK Documentation](https://developer.android.com/ndk)
- [iOS C++ Integration](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_c_into_swift)
- [llama.cpp Mobile Examples](https://github.com/ggerganov/llama.cpp/tree/master/examples)

---

**Module**: 08 - Integration & Applications
**Lesson**: 05 - Mobile Deployment
**Version**: 1.0
**Last Updated**: 2025-11-18
