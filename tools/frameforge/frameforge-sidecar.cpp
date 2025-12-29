#include "frameforge-schema.h"
#include "frameforge-validator.h"
#include "frameforge-json.h"
#include "frameforge-ipc.h"

#include "llama.h"
#include "../../external/whisper/include/whisper.h"
#include "../../common/common.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <cstdio>
#include <cstring>

// System prompt for Llama intent classification
static const char * INTENT_SYSTEM_PROMPT = R"(You are an intent classifier for FrameForge Studio, a professional previsualization software.

Analyze user voice commands and map them to one of these Action Groups:
- CAMERA_CONTROL: Camera movements (pan, tilt, dolly, zoom, lean)
- ACTOR_POSE: Actor positioning and poses
- OBJECT_MGMT: Adding, deleting, moving, or rotating objects
- SHOT_MGMT: Managing shots (save, load)

Extract parameters from the user's natural language input:
- Direction: LEFT, RIGHT, UP, DOWN, FORWARD, BACKWARD
- Degrees: Numeric values for rotation (0-360)
- Speed: Numeric values for movement speed (0-100)
- Target: Names of objects, cameras, or actors
- PoseDescription: Natural language description of a pose

Important rules:
1. If user says "PIN", map it to "PAN" verb
2. If Action Group is ACTOR_POSE, generate a JSON array of joint rotations for the described pose
3. Infer missing subjects when context is clear (e.g., "camera" for camera commands)
4. Return ONLY a valid JSON object with this structure:
{
  "verb": "VERB_NAME",
  "subject": "SubjectName",
  "action_group": "ACTION_GROUP",
  "parameters": {
    "direction": "DIRECTION",
    "degrees": 45.0,
    "speed": 10.0,
    "target": "ObjectName",
    "pose_description": "description",
    "joint_rotations": [{"name": "shoulder_left", "rotation_x": 0, "rotation_y": 45, "rotation_z": 0}]
  }
}

Do not include explanations, only the JSON object.)";

struct frameforge_params {
    std::string whisper_model;
    std::string llama_model;
    std::string audio_file;
    std::string pipe_name = "frameforge_pipe";
    int n_threads = 4;
    bool verbose = false;
};

static void print_usage(const char * argv0) {
    fprintf(stderr, "Usage: %s [options]\n", argv0);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -wm, --whisper-model FNAME  Path to Whisper model file\n");
    fprintf(stderr, "  -lm, --llama-model FNAME    Path to Llama model file\n");
    fprintf(stderr, "  -a,  --audio FILE           Audio file to transcribe (for testing)\n");
    fprintf(stderr, "  -p,  --pipe NAME            Named pipe name (default: frameforge_pipe)\n");
    fprintf(stderr, "  -t,  --threads N            Number of threads (default: 4)\n");
    fprintf(stderr, "  -v,  --verbose              Enable verbose output\n");
    fprintf(stderr, "  -h,  --help                 Show this help message\n");
}

static bool parse_params(int argc, char ** argv, frameforge_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-wm" || arg == "--whisper-model") {
            if (i + 1 < argc) {
                params.whisper_model = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing value for %s\n", arg.c_str());
                return false;
            }
        } else if (arg == "-lm" || arg == "--llama-model") {
            if (i + 1 < argc) {
                params.llama_model = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing value for %s\n", arg.c_str());
                return false;
            }
        } else if (arg == "-a" || arg == "--audio") {
            if (i + 1 < argc) {
                params.audio_file = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing value for %s\n", arg.c_str());
                return false;
            }
        } else if (arg == "-p" || arg == "--pipe") {
            if (i + 1 < argc) {
                params.pipe_name = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing value for %s\n", arg.c_str());
                return false;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                params.n_threads = std::stoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: Missing value for %s\n", arg.c_str());
                return false;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    
    if (params.whisper_model.empty()) {
        fprintf(stderr, "Error: Whisper model path is required\n");
        return false;
    }
    
    if (params.llama_model.empty()) {
        fprintf(stderr, "Error: Llama model path is required\n");
        return false;
    }
    
    return true;
}

// Read WAV file and return PCM audio data
static bool read_wav(const std::string & fname, std::vector<float> & pcmf32, int & sample_rate) {
    // Simple WAV reader - assumes 16-bit PCM
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Failed to open audio file: %s\n", fname.c_str());
        return false;
    }
    
    char buf[256];
    size_t bytes_read = fread(buf, 1, 44, f);  // Read WAV header (44 bytes)
    if (bytes_read != 44) {
        fprintf(stderr, "Error: Failed to read WAV header\n");
        fclose(f);
        return false;
    }

    // Get sample rate from header
    sample_rate = *(int32_t *)(buf + 24);
    
    // Read audio data
    std::vector<int16_t> pcm16;
    int16_t sample;
    while (fread(&sample, sizeof(int16_t), 1, f) == 1) {
        pcm16.push_back(sample);
    }
    
    fclose(f);
    
    // Convert to float
    pcmf32.resize(pcm16.size());
    for (size_t i = 0; i < pcm16.size(); i++) {
        pcmf32[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    }
    
    return true;
}

// Transcribe audio using Whisper
static std::string transcribe_audio(whisper_context * wctx, const std::vector<float> & pcmf32, bool verbose) {
    if (!wctx) {
        return "";
    }
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = verbose;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.translate = false;
    wparams.language = "en";
    wparams.n_threads = 4;
    
    if (whisper_full(wctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stderr, "Error: Failed to process audio\n");
        return "";
    }
    
    std::string text;
    const int n_segments = whisper_full_n_segments(wctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * segment_text = whisper_full_get_segment_text(wctx, i);
        text += segment_text;
    }
    
    return text;
}

// Classify intent using Llama
static std::string classify_intent(llama_context * lctx, llama_model * model, const std::string & user_input, bool verbose) {
    if (!lctx || !model) {
        return "";
    }

    // Get vocab from model
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Build the prompt
    std::string prompt = std::string(INTENT_SYSTEM_PROMPT) + "\n\nUser input: " + user_input + "\n\nJSON output:";

    // Tokenize the prompt
    // First get the required size
    const int                n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
        fprintf(stderr, "Error: Failed to tokenize prompt\n");
        return "";
    }

    if (verbose) {
        fprintf(stderr, "Prompt tokens: %zu\n", tokens.size());
    }
    
    // Evaluate the prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    if (llama_decode(lctx, batch) != 0) {
        fprintf(stderr, "Error: Failed to evaluate prompt\n");
        return "";
    }

    // Generate response
    std::string response;
    const int   max_tokens = 512;

    for (int i = 0; i < max_tokens; i++) {
        auto * logits = llama_get_logits_ith(lctx, -1);
        
        // Simple greedy sampling
        llama_token new_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < llama_vocab_n_tokens(vocab); j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                new_token = j;
            }
        }

        // Check for end of text
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Decode token to text
        char buf[128];
        int  n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            response.append(buf, n);
        }
        
        // Evaluate the new token
        batch = llama_batch_get_one(&new_token, 1);

        if (llama_decode(lctx, batch) != 0) {
            break;
        }

        // Check if we have a complete JSON object
        if (response.find('}') != std::string::npos) {
            break;
        }
    }
    
    if (verbose) {
        fprintf(stderr, "LLM response: %s\n", response.c_str());
    }
    
    return response;
}

int main(int argc, char ** argv) {
    frameforge_params params;
    
    if (!parse_params(argc, argv, params)) {
        return 1;
    }
    
    // Initialize Whisper
    fprintf(stderr, "Loading Whisper model: %s\n", params.whisper_model.c_str());
    whisper_context_params cparams = whisper_context_default_params();
    whisper_context * wctx = whisper_init_from_file_with_params(params.whisper_model.c_str(), cparams);
    if (!wctx) {
        fprintf(stderr, "Error: Failed to load Whisper model\n");
        return 1;
    }
    
    // Initialize Llama
    fprintf(stderr, "Loading Llama model: %s\n", params.llama_model.c_str());
    llama_model_params model_params = llama_model_default_params();
    llama_model *      model        = llama_model_load_from_file(params.llama_model.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: Failed to load Llama model\n");
        whisper_free(wctx);
        return 1;
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    llama_context * lctx            = llama_init_from_model(model, ctx_params);
    if (!lctx) {
        fprintf(stderr, "Error: Failed to create Llama context\n");
        llama_model_free(model);
        whisper_free(wctx);
        return 1;
    }
    
    // Initialize command validator
    frameforge::CommandValidator validator;
    
    // Test mode: process a single audio file
    if (!params.audio_file.empty()) {
        fprintf(stderr, "Processing audio file: %s\n", params.audio_file.c_str());
        
        std::vector<float> pcmf32;
        int sample_rate = 0;
        if (!read_wav(params.audio_file, pcmf32, sample_rate)) {
            llama_free(lctx);
            llama_model_free(model);
            whisper_free(wctx);
            return 1;
        }
        
        fprintf(stderr, "Transcribing audio...\n");
        std::string transcription = transcribe_audio(wctx, pcmf32, params.verbose);
        fprintf(stderr, "Transcription: %s\n", transcription.c_str());
        
        fprintf(stderr, "Classifying intent...\n");
        std::string llm_response = classify_intent(lctx, model, transcription, params.verbose);
        fprintf(stderr, "LLM Response: %s\n", llm_response.c_str());
        
        // Validate the command
        frameforge::Command cmd;
        frameforge::ValidationResult result = validator.validate_json(llm_response, cmd);
        
        if (result.valid) {
            std::string json_output = frameforge::command_to_json(cmd);
            fprintf(stderr, "Valid command:\n%s\n", json_output.c_str());
        } else {
            fprintf(stderr, "Validation failed: %s\n", result.error_message.c_str());
            std::string clarification = validator.generate_clarification_request(result, cmd);
            fprintf(stderr, "Clarification: %s\n", clarification.c_str());
        }
        
        llama_free(lctx);
        llama_model_free(model);
        whisper_free(wctx);
        return 0;
    }
    
    // Server mode: start IPC server
    fprintf(stderr, "Starting IPC server on pipe: %s\n", params.pipe_name.c_str());
    frameforge::IPCServer ipc_server(params.pipe_name);
    
    if (!ipc_server.start()) {
        fprintf(stderr, "Error: Failed to start IPC server\n");
        llama_free(lctx);
        llama_model_free(model);
        whisper_free(wctx);
        return 1;
    }
    
    fprintf(stderr, "FrameForge Sidecar ready. Waiting for commands...\n");
    
    // Main loop
    std::atomic<bool> running(true);
    while (running) {
        // In a real implementation, this would:
        // 1. Receive audio data from the IPC pipe
        // 2. Transcribe with Whisper
        // 3. Classify with Llama
        // 4. Validate the command
        // 5. Send the validated JSON back through the pipe
        
        // For now, just keep the process running
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    ipc_server.stop();
    llama_free(lctx);
    llama_model_free(model);
    whisper_free(wctx);
    
    return 0;
}
