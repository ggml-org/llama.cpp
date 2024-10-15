import Foundation
import llama

let arguments = CommandLine.arguments

// Check that we have at least one argument (the model path)
guard arguments.count > 1 else {
    print("Usage: swift MODEL_PATH [PROMPT] [PARALLEL]")
    exit(1)
}

let modelPath: String = arguments[1]
let prompt: String = arguments.count > 2 ? arguments[2] : "Hello my name is"
let n_parallel: Int = arguments.count > 3 && Int(arguments[3]) != nil ? Int(arguments[3])! : 1

// total length of the sequences including the prompt
let n_len: Int = 32

// init LLM
llama_backend_init()
defer {
    llama_backend_free()
}

let model_params = llama_model_default_params()
guard let model = llama_load_model_from_file(modelPath.cString(using: .utf8), model_params) else {
    print("Failed to load model")
    exit(1)
}
defer {
    llama_free_model(model)
}

var tokens = tokenize(text: prompt, add_bos: true)

let n_kv_req = UInt32(tokens.count) + UInt32((n_len - Int(tokens.count)) * n_parallel)

var context_params = llama_context_default_params()
context_params.n_ctx = n_kv_req
context_params.n_batch = UInt32(max(n_len, n_parallel))
context_params.n_threads = 8
context_params.n_threads_batch = 8

let context = llama_new_context_with_model(model, context_params)
guard context != nil else {
    print("Failed to initialize context")
    exit(1)
}
defer {
    llama_free(context)
}

var sparams = llama_sampler_chain_default_params()

let smpl = llama_sampler_chain_init(sparams)
guard smpl != nil else {
    print("Failed to initialize sampling")
    exit(1)
}
defer {
    llama_sampler_free(smpl)
}

llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.4));
llama_sampler_chain_add(smpl, llama_sampler_init_dist (1234));

let n_ctx = llama_n_ctx(context)

print("\nn_len = \(n_len), n_ctx = \(n_ctx), n_batch = \(context_params.n_batch), n_parallel = \(n_parallel), n_kv_req = \(n_kv_req)\n")

if n_kv_req > n_ctx {
    print("error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", n_kv_req)
    exit(1)
}

var buffer: [CChar] = []
for id: llama_token in tokens {
    print(token_to_piece(token: id, buffer: &buffer) ?? "", terminator: "")
}

print("\n")

var batch = llama_batch_init(max(Int32(tokens.count), Int32(n_parallel)), 0, 1)
defer {
    llama_batch_free(batch)
}

// evaluate the initial prompt
batch.n_tokens = Int32(tokens.count)

for (i, token) in tokens.enumerated() {
    batch.token[i] = token
    batch.pos[i] = Int32(i)
    batch.n_seq_id[i] = 1
    // batch.seq_id[i][0] = 0
    // TODO: is this the proper way to do this?
    if let seq_id = batch.seq_id[i] {
        seq_id[0] = 0
    }
    batch.logits[i] = 0
}

// llama_decode will output logits only for the last token of the prompt
batch.logits[Int(batch.n_tokens) - 1] = 1

if llama_decode(context, batch) != 0 {
    print("llama_decode() failed")
    exit(1)
}

for i in 1 ..< n_parallel {
    llama_kv_cache_seq_cp(context, 0, Int32(i), 0, batch.n_tokens)
}

if n_parallel > 1 {
    print("generating \(n_parallel) sequences ...\n")
}

var streams: [String] = .init(repeating: "", count: n_parallel)
var streamBuffers: [[CChar]] = .init(repeating: [], count: n_parallel)
var i_batch = [Int32](repeating: batch.n_tokens - 1, count: n_parallel)

var n_cur = batch.n_tokens
var n_decode = 0

let t_main_start = ggml_time_us()

while n_cur <= n_len {
    // prepare the next batch
    batch.n_tokens = 0

    // sample the next token for each parallel sequence / stream
    for i in 0 ..< n_parallel {
        if i_batch[i] < 0 {
            // the stream has already finished
            continue
        }

        let new_token_id = llama_sampler_sample(smpl, context, i_batch[i])

        // is it an end of stream? -> mark the stream as finished
        if llama_token_is_eog(model, new_token_id) || n_cur == n_len {
            i_batch[i] = -1
            // print("")
            if n_parallel > 1 {
                print("stream \(i) finished at n_cur = \(n_cur)")
            }

            continue
        }

        let nextStringPiece = token_to_piece(token: new_token_id, buffer: &streamBuffers[i]) ?? ""

        // if there is only one stream, we print immediately to stdout
        if n_parallel == 1 {
            print(nextStringPiece, terminator: "")
        }
        streams[i] += nextStringPiece

        // push this new token for next evaluation
        batch.token[Int(batch.n_tokens)] = new_token_id
        batch.pos[Int(batch.n_tokens)] = n_cur
        batch.n_seq_id[Int(batch.n_tokens)] = 1
        if let seq_id = batch.seq_id[Int(batch.n_tokens)] {
            seq_id[0] = Int32(i)
        }
        batch.logits[Int(batch.n_tokens)] = 1

        i_batch[i] = batch.n_tokens

        batch.n_tokens += 1

        n_decode += 1
    }

    // all streams are finished
    if batch.n_tokens == 0 {
        break
    }

    n_cur += 1

    // evaluate the current batch with the transformer model
    if llama_decode(context, batch) != 0 {
        print("llama_decode() failed")
        exit(1)
    }
}

if n_parallel > 1 {
    print("\n")
    for (i, stream) in streams.enumerated() {
        print("sequence \(i):\n\n\(prompt)\(stream)\n")
    }
}

let t_main_end = ggml_time_us()

print("decoded \(n_decode) tokens in \(String(format: "%.2f", Double(t_main_end - t_main_start) / 1_000_000.0)) s, speed: \(String(format: "%.2f", Double(n_decode) / (Double(t_main_end - t_main_start) / 1_000_000.0))) t/s\n\n")

llama_perf_sampler_print(smpl)
llama_perf_context_print(context)

private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
    let utf8Count = text.utf8.count
    let n_tokens = utf8Count + (add_bos ? 1 : 0)
    let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
    let tokenCount = llama_tokenize(model, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, /*special tokens*/ false)
    var swiftTokens: [llama_token] = []
    for i in 0 ..< tokenCount {
        swiftTokens.append(tokens[Int(i)])
    }
    tokens.deallocate()
    return swiftTokens
}

private func token_to_piece(token: llama_token, buffer: inout [CChar]) -> String? {
    var result = [CChar](repeating: 0, count: 8)
    let nTokens = llama_token_to_piece(model, token, &result, Int32(result.count), 0, false)
    if nTokens < 0 {
        let actualTokensCount = -Int(nTokens)
        result = .init(repeating: 0, count: actualTokensCount)
        let check = llama_token_to_piece(
            model,
            token,
            &result,
            Int32(result.count),
            0,
            false
        )
        assert(check == actualTokensCount)
    } else {
        result.removeLast(result.count - Int(nTokens))
    }
    if buffer.isEmpty, let utfString = String(cString: result + [0], encoding: .utf8) {
        return utfString
    } else {
        buffer.append(contentsOf: result)
        let data = Data(buffer.map { UInt8(bitPattern: $0) })
        if buffer.count >= 4 { // 4 bytes is the max length of a utf8 character so if we're here we need to reset the buffer
            buffer = []
        }
        guard let bufferString = String(data: data, encoding: .utf8) else {
            return nil
        }
        buffer = []
        return bufferString
    }
}
