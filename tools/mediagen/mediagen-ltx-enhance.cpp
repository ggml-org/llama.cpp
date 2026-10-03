#include "mediagen-ltx.h"

#include <cctype>
#include <cstring>

// prompt enhancement: LTX-2 is trained on long captions, short prompts are expanded by the text encoder
// system prompt from LTX-2 (Lightricks, Apache-2.0):
// https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt

static const char * LTX_T2V_SYSTEM_PROMPT =
"You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed\n"
"video generation prompt with specific visuals and integrated audio to guide a text-to-video model.\n"
"\n"
"#### Guidelines\n"
"- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions,\n"
"  actions, camera movement, audio).\n"
"    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.\n"
"        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.\n"
"- Use active language: present-progressive verbs (\"is walking,\" \"speaking\"). If no action specified, describe natural\n"
"  movements.\n"
"- Maintain chronological flow: use temporal connectors (\"as,\" \"then,\" \"while\").\n"
"- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested).\n"
"  Integrate sounds chronologically alongside actions. Be specific (e.g., \"soft footsteps on tile\"), not vague (e.g.,\n"
"  \"ambient sound is present\").\n"
"- Speech (only when requested):\n"
"    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with\n"
"      voice characteristics (e.g., \"The man says in an excited voice: 'You won't believe what I just saw!'\").\n"
"    - Specify language if not English and accent if relevant.\n"
"- Style: Include visual style at the beginning: \"Style: <style>, <rest of prompt>.\" Default to cinematic-realistic if\n"
"  unspecified. Omit if unclear.\n"
"- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).\n"
"- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.\n"
"    - Colors: Use plain terms (\"red dress\"), not intensified (\"vibrant blue,\" \"bright red\").\n"
"    - Lighting: Use neutral descriptions (\"soft overhead light\"), not harsh (\"blinding light\").\n"
"    - Facial features: Use delicate modifiers for subtle features (i.e., \"subtle freckles\").\n"
"\n"
"#### Important notes:\n"
"- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is\n"
"  requested.\n"
"- Camera motion: DO NOT invent camera motion unless requested by the user.\n"
"- Speech: DO NOT modify user-provided character dialogue unless it's a typo.\n"
"- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.\n"
"- Format: DO NOT use phrases like \"The scene opens with...\". Start directly with Style (optional) and chronological\n"
"  scene description.\n"
"- Format: DO NOT start your response with special characters.\n"
"- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.\n"
"- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits\n"
"  or introduce new elements. Add/enhance audio descriptions if missing.\n"
"\n"
"#### Output Format (Strict):\n"
"- Single continuous paragraph in natural language (English).\n"
"- NO titles, headings, prefaces, code fences, or Markdown.\n"
"- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.\n"
"\n"
"Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video\n"
"generation.\n"
"\n"
"#### Example Input: \"A woman at a coffee shop talking on the phone\" Output: Style: realistic with cinematic lighting.\n"
"In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the\n"
"window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone\n"
"to her ear with the other. Ambient cafe sounds fill the space-espresso machine hiss, quiet conversations, gentle\n"
"clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a\n"
"soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd\n"
"love to meet up this weekend. How about Saturday afternoon?' She laughs softly-a genuine chuckle-and shifts in her\n"
"chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully,\n"
"lowering the phone.\n";

// ltx-pipelines clean_response: ascii quotes and dashes, drop leading non-letters
static std::string ltx_clean_response(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size();) {
        const unsigned char c = in[i];
        // utf-8 sequences for the replaced punctuation
        if (c == 0xE2 && i + 2 < in.size()) {
            const unsigned char c1 = in[i + 1], c2 = in[i + 2];
            if (c1 == 0x80 && (c2 == 0x98 || c2 == 0x99 || c2 == 0xB2)) { out += '\''; i += 3; continue; }
            if (c1 == 0x80 && (c2 == 0x9C || c2 == 0x9D)) { out += '"'; i += 3; continue; }
            if (c1 == 0x80 && (c2 == 0x94 || c2 == 0x93)) { out += '-'; i += 3; continue; }
            if (c1 == 0x88 && c2 == 0x92) { out += '-'; i += 3; continue; }
        }
        if (c == 0xC2 && i + 1 < in.size() && (unsigned char) in[i + 1] == 0xA0) { out += ' '; i += 2; continue; }
        out += (char) c;
        i++;
    }
    size_t b = 0;
    while (b < out.size() && !std::isalpha((unsigned char) out[b]) && (unsigned char) out[b] < 0x80) {
        b++;
    }
    out = out.substr(b);
    size_t e = out.find_last_not_of(" \t\r\n");
    return e == std::string::npos ? out : out.substr(0, e + 1);
}

bool ltx_text_encoder::enhance(const std::string & prompt, uint32_t seed, int max_new_tokens, std::string & out) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the model's chat template merges the system prompt into the user turn
    std::string user = std::string("user prompt: ") + prompt;
    llama_chat_message msgs[2] = {
        { "system", LTX_T2V_SYSTEM_PROMPT },
        { "user",   user.c_str() },
    };
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(strlen(LTX_T2V_SYSTEM_PROMPT) + user.size() + 512);
    int32_t n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int32_t) buf.size());
    if (n < 0) {
        MG_ERR("%s: failed to apply the chat template\n", __func__);
        return false;
    }
    if (n > (int32_t) buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int32_t) buf.size());
    }
    const std::vector<llama_token> tokens = ltx_tokenize(vocab, std::string(buf.data(), n), true);
    const int n_tok = (int) tokens.size();
    if (n_tok == 0) {
        return false;
    }
    if (n_tok + max_new_tokens > n_ctx) {
        MG_ERR("%s: prompt too long for enhancement (%d tokens)\n", __func__, n_tok);
        return false;
    }

    // same context as encode(), in logits mode
    llama_set_embeddings(lctx, false);
    llama_memory_clear(llama_get_memory(lctx), true);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    for (int i = 0; i < n_tok; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i == n_tok - 1;
    }
    batch.n_tokens = n_tok;

    bool ok = true;
    if (llama_decode(lctx, batch) != 0) {
        MG_ERR("%s: llama_decode failed on the prompt\n", __func__);
        ok = false;
    }

    llama_sampler * smpl = nullptr;
    if (ok) {
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }

    std::string gen;
    int pos = n_tok;
    for (int i = 0; ok && i < max_new_tokens; i++) {
        const llama_token id = llama_sampler_sample(smpl, lctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char piece[256];
        const int np = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, true);
        if (np > 0) {
            gen.append(piece, np);
        }
        batch.token[0]     = id;
        batch.pos[0]       = pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        batch.n_tokens     = 1;
        if (llama_decode(lctx, batch) != 0) {
            MG_ERR("%s: llama_decode failed during generation\n", __func__);
            ok = false;
        }
    }
    if (smpl) {
        llama_sampler_free(smpl);
    }
    llama_batch_free(batch);

    // back to embeddings mode
    llama_set_embeddings(lctx, true);
    llama_memory_clear(llama_get_memory(lctx), true);

    if (!ok) {
        return false;
    }
    out = ltx_clean_response(gen);
    if (out.empty()) {
        out = prompt;
    }
    return true;
}
