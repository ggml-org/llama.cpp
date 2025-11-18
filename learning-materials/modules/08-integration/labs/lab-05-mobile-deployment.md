# Lab 8.5: Mobile Deployment

**Estimated Time**: 4-5 hours
**Difficulty**: Advanced
**Prerequisites**: Lesson 8.5, Mobile development experience

## Objective

Deploy llama.cpp to mobile devices (Android and/or iOS) with optimized performance, efficient resource usage, and native UI integration.

## Part 1: Android Deployment (Choose Android or iOS)

### Android Setup (2 hours)

#### Task 1.1: Project Setup

Create Android project structure:

```
LLaMobileApp/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/llama/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── LlamaModel.kt
│   │   │   │   └── ChatViewModel.kt
│   │   │   ├── cpp/
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── llama_jni.cpp
│   │   │   │   └── llama.cpp/ (submodule)
│   │   │   └── res/
│   │   └── build.gradle
│   └── build.gradle
```

#### Task 1.2: JNI Bridge

Implement JNI wrapper:

```cpp
// llama_jni.cpp
#include <jni.h>
#include "llama.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_llama_LlamaModel_nativeInit(
    JNIEnv* env,
    jobject obj,
    jstring model_path,
    jint n_ctx
) {
    // TODO: Load model and create context
    // Return pointer as jlong
}

JNIEXPORT jstring JNICALL
Java_com_example_llama_LlamaModel_nativeGenerate(
    JNIEnv* env,
    jobject obj,
    jlong ctx_ptr,
    jstring prompt,
    jint max_tokens
) {
    // TODO: Generate text
    // Return result as jstring
}

JNIEXPORT void JNICALL
Java_com_example_llama_LlamaModel_nativeFree(
    JNIEnv* env,
    jobject obj,
    jlong ctx_ptr
) {
    // TODO: Free resources
}

} // extern "C"
```

#### Task 1.3: Kotlin Wrapper

```kotlin
// LlamaModel.kt
package com.example.llama

class LlamaModel {
    private var nativeHandle: Long = 0

    companion object {
        init {
            System.loadLibrary("llama-jni")
        }
    }

    fun load(modelPath: String, nCtx: Int = 2048): Boolean {
        // TODO: Load model
        nativeHandle = nativeInit(modelPath, nCtx)
        return nativeHandle != 0L
    }

    fun generate(prompt: String, maxTokens: Int = 256): String {
        // TODO: Generate text
        return nativeGenerate(nativeHandle, prompt, maxTokens)
    }

    fun free() {
        // TODO: Free resources
        if (nativeHandle != 0L) {
            nativeFree(nativeHandle)
            nativeHandle = 0
        }
    }

    // Native methods
    private external fun nativeInit(modelPath: String, nCtx: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, maxTokens: Int): String
    private external fun nativeFree(handle: Long)
}
```

#### Task 1.4: UI with Jetpack Compose

```kotlin
// MainActivity.kt
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    var inputText by remember { mutableStateOf("") }
    val messages by viewModel.messages.collectAsState()

    Column(modifier = Modifier.fillMaxSize()) {
        // Messages list
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(16.dp)
        ) {
            items(messages) { message ->
                MessageBubble(message)
            }
        }

        // Input
        Row(modifier = Modifier.padding(16.dp)) {
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier.weight(1f)
            )
            Button(onClick = {
                viewModel.sendMessage(inputText)
                inputText = ""
            }) {
                Text("Send")
            }
        }
    }
}
```

## Part 2: iOS Deployment (Alternative)

### iOS Setup (2 hours)

#### Task 2.1: Swift Wrapper

```swift
// LlamaModel.swift
import Foundation

class LlamaModel {
    private var context: OpaquePointer?
    private var model: OpaquePointer?

    func load(modelPath: String) -> Bool {
        // TODO: Load model using llama.cpp C API
        var params = llama_model_default_params()
        model = llama_load_model_from_file(modelPath, params)

        guard model != nil else { return false }

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = 2048
        context = llama_new_context_with_model(model, ctxParams)

        return context != nil
    }

    func generate(prompt: String, maxTokens: Int = 256) -> String {
        // TODO: Implement generation
        guard let context = context else { return "" }

        // Tokenize, evaluate, sample loop
        // Return generated text
    }

    deinit {
        // Cleanup
        if let context = context {
            llama_free(context)
        }
        if let model = model {
            llama_free_model(model)
        }
    }
}
```

#### Task 2.2: SwiftUI Interface

```swift
// ContentView.swift
struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var inputText = ""

    var body: some View {
        VStack {
            // Messages
            ScrollView {
                LazyVStack(alignment: .leading) {
                    ForEach(viewModel.messages) { message in
                        MessageView(message: message)
                    }
                }
            }

            // Input
            HStack {
                TextField("Message", text: $inputText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())

                Button("Send") {
                    viewModel.send(inputText)
                    inputText = ""
                }
            }
            .padding()
        }
    }
}
```

## Part 3: Optimization (60 min)

### Task 3.1: Model Optimization

Optimize for mobile:

```python
# optimize_model.py
"""
Optimize model for mobile deployment.

Steps:
1. Quantize to Q4_K_M or smaller
2. Reduce context window if needed
3. Test on device
"""

from llama_cpp import Llama

def optimize_for_mobile(
    input_model: str,
    output_model: str,
    target_size_mb: int = 2000
):
    # TODO: Implement model optimization
    # - Choose appropriate quantization
    # - Verify size constraints
    # - Test quality
    pass
```

### Task 3.2: Performance Monitoring

```kotlin
// PerformanceMonitor.kt
class PerformanceMonitor {
    fun measureInference(block: () -> String): InferenceMetrics {
        val startTime = System.currentTimeMillis()
        val startMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()

        val result = block()

        val endTime = System.currentTimeMillis()
        val endMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()

        return InferenceMetrics(
            durationMs = endTime - startTime,
            memoryUsedMb = (endMemory - startMemory) / (1024 * 1024),
            tokensGenerated = result.split(" ").size,
            tokensPerSecond = /* calculate */
        )
    }
}
```

## Part 4: Testing (30 min)

### Device Testing Checklist

- [ ] Model loads successfully
- [ ] Inference completes without errors
- [ ] Response time acceptable (< 5s for 256 tokens)
- [ ] Memory usage within limits
- [ ] No thermal throttling
- [ ] Battery drain acceptable
- [ ] UI remains responsive during generation
- [ ] Works offline

### Automated Tests

```kotlin
// LlamaModelTest.kt
@Test
fun testModelLoading() {
    val model = LlamaModel()
    val loaded = model.load(getModelPath())
    assertTrue(loaded)
}

@Test
fun testGeneration() {
    val model = LlamaModel()
    model.load(getModelPath())
    val result = model.generate("Hello", maxTokens = 50)
    assertFalse(result.isEmpty())
}

@Test
fun testPerformance() {
    val model = LlamaModel()
    model.load(getModelPath())

    val startTime = System.currentTimeMillis()
    model.generate("Test prompt", maxTokens = 100)
    val duration = System.currentTimeMillis() - startTime

    // Should complete in reasonable time
    assertTrue(duration < 10000) // 10 seconds
}
```

## Challenges

### Challenge 1: Streaming on Mobile
Implement token-by-token streaming with UI updates.

### Challenge 2: Background Processing
Continue generation when app goes to background.

### Challenge 3: Model Switching
Allow users to download and switch between models.

### Challenge 4: On-Device Training
Implement simple fine-tuning on device.

### Challenge 5: Multi-Modal
Add image input support.

## Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Model Load Time | < 3s | < 1s |
| First Token Latency | < 500ms | < 200ms |
| Tokens/Second | > 5 | > 10 |
| Memory Usage | < 4GB | < 2GB |
| Battery Drain | < 10%/hour | < 5%/hour |

## Success Criteria

- [X] App builds and runs on device
- [X] Model loads successfully
- [X] Text generation working
- [X] UI responsive and polished
- [X] Performance targets met
- [X] Memory usage acceptable
- [X] Tests passing

## Submission

Submit:
1. Complete app source code
2. APK or IPA file
3. Performance test results
4. Screenshots/video demo
5. Deployment documentation

---

**Lab**: 8.5 - Mobile Deployment
**Module**: 08 - Integration & Applications
**Version**: 1.0
