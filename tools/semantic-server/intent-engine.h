#pragma once

#include "command-schema.h"
#include "command-validator.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <string>
#include <memory>

namespace semantic_server {

// Intent classification system prompt
static const char * INTENT_SYSTEM_PROMPT = 
R"(You are an AI assistant for FrameForge Studio, a professional 3D pre-visualization software for film and television.

Your task is to analyze user input and convert it into structured JSON commands.

Action Groups:
- CAMERA_CONTROL: Camera movements and adjustments (pan, tilt, zoom, dolly, etc.)
- ACTOR_POSE: Actor/character pose modifications
- OBJECT_MGMT: Adding, deleting, selecting objects in the scene
- SHOT_MGMT: Shot creation and editing

Valid Verbs by Action Group:
CAMERA_CONTROL: PAN, TILT, LEAN, ROLL, DOLLY, TRUCK, PEDESTAL, ZOOM, FOCUS
ACTOR_POSE: MODIFY
OBJECT_MGMT: ADD, DELETE, SELECT
SHOT_MGMT: SHOT, CUT

Instructions:
1. Identify the Action Group from the user's input
2. Map to the most appropriate Verb (use fuzzy matching - e.g., "PIN" -> "PAN")
3. Extract parameters from the input
4. For ACTOR_POSE with MODIFY verb, if pose description is provided, generate a JSON array of joint rotations
5. Return ONLY valid JSON with this structure:
{
  "verb": "VERB_NAME",
  "action_group": "ACTION_GROUP_NAME",
  "parameters": {
    "param1": value1,
    "param2": value2
  }
}

Examples:
Input: "Pan left 30 degrees"
Output: {"verb":"PAN","action_group":"CAMERA_CONTROL","parameters":{"direction":"LEFT","degrees":30}}

Input: "PIN LEFT"
Output: {"verb":"PAN","action_group":"CAMERA_CONTROL","parameters":{"direction":"LEFT"}}

Input: "Add a chair"
Output: {"verb":"ADD","action_group":"OBJECT_MGMT","parameters":{"object_type":"chair"}}

Input: "Make the actor sit down"
Output: {"verb":"MODIFY","action_group":"ACTOR_POSE","parameters":{"subject":"actor","pose_description":"sitting"}}

Input: "Dolly in slowly"
Output: {"verb":"DOLLY","action_group":"CAMERA_CONTROL","parameters":{"direction":"IN","speed":0.5}}

Return ONLY the JSON object, no other text.)";

class IntentEngine {
public:
    IntentEngine(llama_context * ctx, llama_model * model) 
        : ctx(ctx), model(model), vocab(llama_model_get_vocab(model)) {
        validator = std::make_unique<CommandValidator>();
        
        // Initialize sampler with low temperature for deterministic output
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        smpl = llama_sampler_chain_init(sparams);
        
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.1f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    
    ~IntentEngine() {
        if (smpl) {
            llama_sampler_free(smpl);
        }
    }
    
    // Process user input and return validated command JSON
    ValidationResult process_input(const std::string & user_input) {
        // Build the prompt with system instructions and user input
        std::string prompt = std::string(INTENT_SYSTEM_PROMPT) + "\n\nUser Input: " + user_input + "\nJSON Output:";
        
        // Generate response from LLM
        std::string llm_response = generate_response(prompt);
        
        // Parse JSON response
        json command;
        try {
            command = json::parse(llm_response);
        } catch (const json::parse_error & e) {
            // Try to extract JSON from response if it's embedded in text
            command = extract_json_from_text(llm_response);
            if (command.is_null()) {
                return ValidationResult::error("Failed to parse LLM response as JSON: " + std::string(e.what()));
            }
        }
        
        // Check if verb needs fuzzy matching correction
        if (command.contains("verb")) {
            std::string verb_str = command["verb"].get<std::string>();
            std::string suggested = validator->suggest_verb(verb_str);
            if (!suggested.empty() && suggested != verb_str) {
                command["verb"] = suggested;
            }
        }
        
        // Validate the command against schema
        return validator->validate(command);
    }

private:
    llama_context * ctx;
    llama_model * model;
    const llama_vocab * vocab;
    llama_sampler * smpl;
    std::unique_ptr<CommandValidator> validator;
    
    // Generate response from the LLM
    std::string generate_response(const std::string & prompt) {
        // Tokenize the prompt
        const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        if (n_prompt <= 0) {
            return "{}"; // Return empty object on error
        }
        
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            return "{}"; // Return empty object on error
        }
        
        // Prepare batch for prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        
        // Decode prompt
        if (llama_decode(ctx, batch)) {
            return "{}"; // Return empty object on error
        }
        
        // Generate response tokens
        std::string response;
        const int n_predict = 512; // Maximum tokens to generate
        
        for (int n_generated = 0; n_generated < n_predict; n_generated++) {
            // Sample next token
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
            
            // Check for end of generation
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }
            
            // Convert token to text
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                response.append(buf, n);
            }
            
            // Check if we have a complete JSON object
            if (response.find('}') != std::string::npos) {
                // Try to parse to see if it's complete
                try {
                    json test = extract_json_from_text(response);
                    if (!test.is_null() && test.is_object()) {
                        break; // We have a complete JSON object
                    }
                } catch (...) {
                    // Continue generating
                }
            }
            
            // Prepare next batch
            batch = llama_batch_get_one(&new_token_id, 1);
            
            if (llama_decode(ctx, batch)) {
                break;
            }
        }
        
        return response;
    }
    
    // Extract JSON object from text that may contain additional content
    json extract_json_from_text(const std::string & text) {
        // Find the first '{' and last '}'
        size_t start = text.find('{');
        size_t end = text.rfind('}');
        
        if (start != std::string::npos && end != std::string::npos && start < end) {
            std::string json_str = text.substr(start, end - start + 1);
            try {
                return json::parse(json_str);
            } catch (...) {
                return json();
            }
        }
        
        return json();
    }
};

} // namespace semantic_server
