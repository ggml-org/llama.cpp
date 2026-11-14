// JNI surface for the native engine.
//
// Deliberately tiny: the Java side only stores a pointer and forwards calls.
// Everything that could be called inference (tokenizer, conditioning, segments,
// depth heads, codec, mixing) lives in m5_engine.cpp.
#include "m5_engine.h"

#include <jni.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

JavaVM * g_vm = nullptr;

std::string to_string(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string out = chars ? chars : "";
    if (chars) env->ReleaseStringUTFChars(value, chars);
    return out;
}

jstring to_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

void throw_illegal(JNIEnv * env, const std::string & message) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type) env->ThrowNew(type, message.c_str());
}

// One JSON line, same shape as the CLI, so the Java side parses one thing.
std::string stats_json(const m5::Stats & stats) {
    char buffer[512];
    snprintf(buffer, sizeof(buffer),
             "{\"audio_s\":%.4f,\"elapsed_s\":%.4f,\"rtf\":%.4f,\"backbone_s\":%.3f,"
             "\"depth_s\":%.3f,\"codec_s\":%.3f,\"first_pcm_s\":%.3f,\"frames\":%d,\"segments\":%d}",
             stats.audio, stats.elapsed, stats.rtf, stats.backbone, stats.depth, stats.codec,
             stats.first_pcm, stats.frames, stats.segments);
    return buffer;
}

// Streams PCM into the listener.
//
// The codec worker owns this call, so the thread has to be attached before it
// may touch Java objects. The detach is not optional: a native thread left
// attached keeps the VM alive and the process never exits. One attach pair per
// emitted chunk costs a few microseconds against a ~15 ms chunk.
struct Listener {
    jobject target = nullptr;
    jmethodID method = nullptr;

    void operator()(const float * samples, size_t count) {
        if (!target || !method || count == 0 || !g_vm) return;
        JNIEnv * env = nullptr;
        bool attached = false;
        if (g_vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
            if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
            attached = true;
        }
        jfloatArray array = env->NewFloatArray((jsize) count);
        if (array) {
            env->SetFloatArrayRegion(array, 0, (jsize) count, samples);
            env->CallVoidMethod(target, method, array, (jint) count);
            env->DeleteLocalRef(array);
            if (env->ExceptionCheck()) env->ExceptionClear();  // never abort a render on a UI error
        }
        if (attached) g_vm->DetachCurrentThread();
    }
};

Listener * prepare_listener(JNIEnv * env, jobject listener) {
    if (!listener) return nullptr;
    auto * prepared = new Listener();
    if (!g_vm && env->GetJavaVM(&g_vm) != JNI_OK) { delete prepared; return nullptr; }
    if (!g_vm) { delete prepared; return nullptr; }
    prepared->target = env->NewGlobalRef(listener);
    jclass type = env->GetObjectClass(listener);
    prepared->method = env->GetMethodID(type, "onPcm", "([FI)V");
    env->DeleteLocalRef(type);
    if (!prepared->method) {
        env->DeleteGlobalRef(prepared->target);
        delete prepared;
        return nullptr;
    }
    return prepared;
}

void release_listener(JNIEnv * env, Listener * listener) {
    if (!listener) return;
    if (listener->target) env->DeleteGlobalRef(listener->target);
    delete listener;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_m5_offline_NativeEngine_nativeCreate(JNIEnv * env, jclass, jstring model_dir, jstring backbone,
                                              jstring depth, jstring codec, jstring work_dir,
                                              jint voice, jint threads, jint first_frames,
                                              jint chunk_frames, jlong seed, jboolean ring,
                                              jstring voices_path, jstring voice_name,
                                              jstring adsp_dir) {
    // The DSP skel (libggml-htp-v73.so) is searched on ADSP_LIBRARY_PATH, and
    // the stage executables resolve libllama/libggml over LD_LIBRARY_PATH; the
    // app cannot setenv from Java, so the JNI entry does it before any stage
    // process is spawned. The library dir holds both sets of .so files.
    const std::string adsp = to_string(env, adsp_dir);
    if (!adsp.empty()) {
        setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 0);
        const char * old_path = getenv("LD_LIBRARY_PATH");
        const std::string lib_path = old_path && *old_path ? adsp + ":" + old_path : adsp;
        setenv("LD_LIBRARY_PATH", lib_path.c_str(), 1);
    }

    m5::Options options;
    options.model_dir = to_string(env, model_dir);
    options.backbone_cmd = to_string(env, backbone);
    options.depth_cmd = to_string(env, depth);
    options.codec_model = to_string(env, codec);
    options.workdir = to_string(env, work_dir);
    options.voice = voice;
    options.threads = threads;
    options.first_frames = first_frames;
    options.chunk_frames = chunk_frames;
    options.seed = (long) seed;
    options.ring = ring == JNI_TRUE;
    options.voices_path = to_string(env, voices_path);
    options.voice_name = to_string(env, voice_name);

    auto * engine = new m5::Engine();
    std::string error;
    if (!engine->open(options, error)) {
        delete engine;
        throw_illegal(env, "native engine: " + error);
        return 0;
    }
    return (jlong) (intptr_t) engine;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_m5_offline_NativeEngine_nativeSetVoice(JNIEnv * env, jclass, jlong handle, jstring selector) {
    auto * engine = reinterpret_cast<m5::Engine *>((intptr_t) handle);
    if (!engine) { throw_illegal(env, "native engine is closed"); return JNI_FALSE; }
    std::string error;
    if (!engine->set_voice(to_string(env, selector), error)) return JNI_FALSE;
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_m5_offline_NativeEngine_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    delete reinterpret_cast<m5::Engine *>((intptr_t) handle);
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_m5_offline_NativeEngine_nativeEncode(JNIEnv * env, jclass, jlong handle, jstring text) {
    auto * engine = reinterpret_cast<m5::Engine *>((intptr_t) handle);
    if (!engine) { throw_illegal(env, "native engine is closed"); return nullptr; }
    const std::vector<std::vector<long>> segments = engine->encode_segments(to_string(env, text));
    size_t total = 0;
    for (const auto & segment : segments) total += segment.size() + 1;
    jlongArray out = env->NewLongArray((jsize) total);
    if (!out) return nullptr;
    std::vector<jlong> flat;
    flat.reserve(total);
    for (const auto & segment : segments) {
        for (long id : segment) flat.push_back((jlong) id);
        flat.push_back(-1);  // segment separator
    }
    env->SetLongArrayRegion(out, 0, (jsize) flat.size(), flat.data());
    return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_m5_offline_NativeEngine_nativeRender(JNIEnv * env, jclass, jlong handle, jstring text,
                                              jstring wav_path, jobject listener) {
    auto * engine = reinterpret_cast<m5::Engine *>((intptr_t) handle);
    if (!engine) { throw_illegal(env, "native engine is closed"); return nullptr; }

    Listener * prepared = prepare_listener(env, listener);
    m5::PcmSink sink;
    if (prepared) {
        Listener * target = prepared;
        sink = [target](const float * samples, size_t count) { (*target)(samples, count); };
    }

    engine->clear_stop();
    m5::Stats stats;
    std::string error;
    const bool ok = engine->render_text(to_string(env, text), to_string(env, wav_path), sink, stats, error);
    release_listener(env, prepared);
    if (!ok) return to_jstring(env, "{\"error\":\"" + error + "\"}");
    return to_jstring(env, stats_json(stats));
}

extern "C" JNIEXPORT void JNICALL
Java_com_m5_offline_NativeEngine_nativeStop(JNIEnv *, jclass, jlong handle) {
    auto * engine = reinterpret_cast<m5::Engine *>((intptr_t) handle);
    if (engine) engine->stop();
}
