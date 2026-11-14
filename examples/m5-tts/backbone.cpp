#include "ggml-backend.h"
#include "llama.h"

#include "m5_depth_heads.h"
#include "m5_ring.h"

#include <chrono>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Compiling the ring path into the binary costs ~40% on the pipe path for the
// SM6650 even though the code never runs (measured: 62 ms vs 42 ms per depth
// frame, both with `use_ring == false`). Only the ring build pays that, so the
// transport is a build-time choice rather than a runtime flag.
#ifndef M5_STAGE_RING
#define M5_STAGE_RING 0
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>

// The input contains contiguous little-endian FP32 rows. Decode the prefix as
// one batch, then decode one row per frame and write the last hidden each time.
//
// Two transports: the original stdin/stdout framing, and a shared-memory ring
// the parent hands over as `--ring-fd N`. The ring removes two context switches
// and two copies from every call, which is what the depth stage needs since it
// makes sixteen calls per audio frame.
int main(int argc, char ** argv) {
    int ring_fd = -1;
    std::vector<const char *> args;
    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--ring-fd") == 0 && i + 1 < argc) {
            ring_fd = std::atoi(argv[++i]);
            continue;
        }
        args.push_back(argv[i]);
    }

    const bool depth = args.size() == 4 && std::strcmp(args[3], "--depth-pipe") == 0;
    const bool pipe = depth || (args.size() == 4 && std::strcmp(args[3], "--pipe") == 0);
    // --depth-chain: the whole sixteen-book head chain runs inside this
    // process. One pipe round trip per audio frame instead of sixteen.
    //   args = [argv0, model.gguf, CPU|HTP0, --depth-chain, depth-weights.f32]
    const bool chain = args.size() == 5 && std::strcmp(args[3], "--depth-chain") == 0;
    if (args.size() != 6 && !pipe && !chain) {
        std::fprintf(stderr, "usage: %s model.gguf CPU|HTP0 input.f32 prefix_rows output.f32\n", argv[0]);
        std::fprintf(stderr, "       %s model.gguf CPU|HTP0 --pipe|--depth-pipe\n", argv[0]);
        std::fprintf(stderr, "       %s model.gguf CPU|HTP0 --depth-chain depth-weights.f32\n", argv[0]);
        return 2;
    }

    try {
        size_t parsed = 0;
        const auto prefix = chain ? 2ul : (pipe ? (depth ? 2ul : 2048ul) : std::stoul(argv[4], &parsed));
        if (!pipe && !chain && (parsed != std::strlen(argv[4]) || prefix == 0 || prefix > 2048)) {
            throw std::runtime_error("invalid prefix length");
        }

        ggml_backend_load_all();
        llama_backend_init();
        auto model_params = llama_model_default_params();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(args[2]), nullptr};
        if (!devices[0]) {
            throw std::runtime_error("requested backend is unavailable");
        }
        model_params.devices = devices;
        model_params.n_gpu_layers = std::strcmp(args[2], "CPU") == 0 ? 0 : -1;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(args[1], model_params), llama_model_free);
        if (!model || llama_model_n_embd_inp(model.get()) != 512 || llama_model_n_embd_out(model.get()) != 512) {
            throw std::runtime_error("expected a 512-wide M5 backbone");
        }

        constexpr int width = 512;
        constexpr auto row_bytes = width * sizeof(float);
        std::ifstream input_file;
        std::ofstream output_file;
        int rows = depth ? 16 : 2048;
        if (!pipe && !chain) {
            input_file.open(argv[3], std::ios::binary | std::ios::ate);
            const auto bytes = input_file.tellg();
            if (bytes < 0 || bytes % row_bytes != 0 || bytes / row_bytes > 2048 || bytes / row_bytes < prefix) {
                throw std::runtime_error("invalid embedding file size");
            }
            rows = static_cast<int>(bytes / row_bytes);
            input_file.seekg(0);
            output_file.open(argv[5], std::ios::binary);
        }
        std::istream & input = (pipe || chain) ? std::cin : input_file;
        std::ostream & output = (pipe || chain) ? std::cout : output_file;

        m5::ring::Server ring;
        const bool use_ring = ring_fd >= 0 && ring.attach(ring_fd);
        if (ring_fd >= 0 && !use_ring) {
            std::fprintf(stderr, "cannot attach to the shared ring\n");
            return 1;
        }
        if (use_ring && ring.width() != width) {
            std::fprintf(stderr, "ring width %d does not match %d\n", ring.width(), width);
            return 1;
        }

        auto params = llama_context_default_params();
        params.embeddings = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        params.type_k = GGML_TYPE_F32;
        params.type_v = GGML_TYPE_F32;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.n_ctx = (depth || chain) ? 256 : 2048;
        params.n_batch = prefix;
        params.n_ubatch = prefix;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), params), llama_free);
        if (!context) {
            throw std::runtime_error("cannot initialize backbone context");
        }

        if (chain) {
            // ---- --depth-chain ------------------------------------------------
            // One request per audio frame: 66 rows in (hidden + JavaRandom noise
            // + control), 2 rows out (codes + next hidden). The sixteen-book
            // chain from m5_depth_heads.h runs entirely inside this process.
            using namespace m5::heads;
            constexpr int kWidth_ = kWidth;
            constexpr uint32_t kChainRows = 66;  // 512 + 32816 + ctrl, rounded up
            static_assert(kChainRows * kWidth >= kWidth + kNoiseBooks * kVocab + 1, "chain message");

            int weights_fd = open(args[4], O_RDONLY);
            if (weights_fd < 0) throw std::runtime_error("cannot open depth weights");
            struct stat weights_st {};
            if (fstat(weights_fd, &weights_st) != 0) {
                ::close(weights_fd);
                throw std::runtime_error("cannot stat depth weights");
            }
            const size_t want_bytes = layout().audio * sizeof(float) +
                                      (size_t) kBooks * kVocab * kWidth * sizeof(float);
            if ((size_t) weights_st.st_size < want_bytes) {
                ::close(weights_fd);
                throw std::runtime_error("depth weights file is too small");
            }
            void * mapped = mmap(nullptr, (size_t) weights_st.st_size, PROT_READ, MAP_PRIVATE, weights_fd, 0);
            ::close(weights_fd);
            if (mapped == MAP_FAILED) throw std::runtime_error("cannot map depth weights");
            const float * w = static_cast<const float *>(mapped);
            std::vector<float16_t> w16(f16_halves());
            build_f16(w, w16.data(), w16.size());
            // Per-row int8 mirror of the project blocks (half the f16 weight
            // traffic); default on, M5_HEADS_INT8=0 keeps the f16 path.
            std::vector<int8_t> w8;
            std::vector<float> w8s;
            const char * env8 = std::getenv("M5_HEADS_INT8");
            if (!env8 || std::strcmp(env8, "0") != 0) {
                w8.assign(int8_rows() * kWidth, (int8_t) 0);
                w8s.assign(int8_rows(), 1.0f);
                build_int8(w, w8.data(), w8s.data(), int8_rows());
            }

            std::vector<float> embeddings(2 * kWidth_);
            std::vector<llama_pos> positions(2);
            std::vector<int32_t> sequence_counts(2, 1);
            llama_seq_id sequence = 0;
            std::vector<llama_seq_id *> sequence_ids(2, &sequence);
            std::vector<int8_t> requested(2);
            std::vector<float> message((size_t) kChainRows * kWidth_);
            float reply[2 * kWidth_];
            int start = 0;
            int frames = 0;

            auto decode = [&](const float * data, int n, float * hidden) -> bool {
                for (int i = 0; i < n; i++) {
                    positions[i] = start + i;
                    requested[i] = i == n - 1;
                }
                std::memcpy(embeddings.data(), data, (size_t) n * kWidth_ * sizeof(float));
                llama_batch batch{};
                batch.n_tokens = n;
                batch.embd = embeddings.data();
                batch.pos = positions.data();
                batch.n_seq_id = sequence_counts.data();
                batch.seq_id = sequence_ids.data();
                batch.logits = requested.data();
                if (llama_decode(context.get(), batch) != 0) {
                    throw std::runtime_error("depth decode failed");
                }
                const float * h = llama_get_embeddings_ith(context.get(), -1);
                if (!h) throw std::runtime_error("cannot read depth hidden");
                std::memcpy(hidden, h, (size_t) kWidth_ * sizeof(float));
                start += n;
                return true;
            };

            for (;;) {
                uint32_t count = 0;
                if (!input.read(reinterpret_cast<char *>(&count), sizeof(count))) {
                    if (input.eof() && input.gcount() == 0) break;
                    throw std::runtime_error("incomplete chain message header");
                }
                if (count == 0) {
                    // Frame-state reset without a frame (engine shutdown or
                    // probe); matches PipeProcess::reset and stays fire-and-forget.
                    llama_memory_clear(llama_get_memory(context.get()), false);
                    start = 0;
                    continue;
                }
                if (count != kChainRows) {
                    throw std::runtime_error("unexpected chain message size");
                }
                if (!input.read(reinterpret_cast<char *>(message.data()),
                                (size_t) count * kWidth_ * sizeof(float))) {
                    throw std::runtime_error("cannot read chain message");
                }
                const float * hidden = message.data();
                const float * noise = message.data() + kWidth_;
                const bool allow_eos =
                    message.data()[kWidth_ + kNoiseBooks * kVocab] != 0.0f;
                llama_memory_clear(llama_get_memory(context.get()), false);
                start = 0;
                long codes[kBooks];
                float next[kWidth_];
                const auto started = std::chrono::steady_clock::now();
                m5::heads::chain(w, w16.data(), hidden, noise, allow_eos, codes, next, decode,
                                 w8.empty() ? nullptr : w8.data(),
                                 w8.empty() ? nullptr : w8s.data());
                const double core_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                std::memset(reply, 0, sizeof(reply));
                for (int j = 0; j < kBooks; j++) reply[j] = (float) codes[j];
                std::memcpy(reply + kWidth_, next, sizeof(float) * kWidth_);
                if (!output.write(reinterpret_cast<const char *>(reply), sizeof(reply)).flush()) {
                    throw std::runtime_error("cannot write chain reply");
                }
                std::fprintf(stderr, "m5-depth frame=%d core_ms=%.3f\n", frames++, core_ms);
            }
            std::fprintf(stderr, "m5-depth frames=%d (chain)\n", frames);
            return 0;
        }

        std::vector<float> embeddings(prefix * width);
        std::vector<llama_pos> positions(prefix);
        std::vector<int32_t> sequence_counts(prefix, 1);
        llama_seq_id sequence = 0;
        std::vector<llama_seq_id *> sequence_ids(prefix, &sequence);
        std::vector<int8_t> requested(prefix);
        // Pipe messages contain a uint32 row count followed by contiguous FP32 rows.
        double prefill_ms = 0;
        double decode_ms = 0;
        double wait_ms = 0;
        int decode_calls = 0;
        int wait_calls = 0;
        int frames = 0;
        // Prefix-cache state for the -1 prefill message (pipe transport only).
        int cached_shared = 0;
        uint32_t cached_hash = 0;
#ifdef M5_STAGE_TIMING
        double decode_api_ms = 0;
        double embedding_api_ms = 0;
        int profiled_calls = 0;
#endif

        // The transport is a compile-time choice. Leaving the ring branch in the
        // pipe build cost 20 ms per depth frame on the SM6650 even though it
        // never ran: the extra branch changed the code generated for the
        // read/decode loop. `if constexpr` keeps both paths specialised.
        auto serve = [&](auto ring_mode) {
        constexpr bool kRing = decltype(ring_mode)::value;
        for (int start = 0; pipe || start < rows;) {
            uint32_t count = start == 0 ? static_cast<uint32_t>(prefix) : 1;
            const float * message = nullptr;
            const auto wait_started = std::chrono::steady_clock::now();
            if constexpr (kRing) {
                int got = 0;
                if (ring.next(got) != 0) break;   // parent is shutting down
                if (got == 0) { count = 0; }
                else { count = static_cast<uint32_t>(got); message = ring.request(); }
            } else {
                if (pipe && !input.read(reinterpret_cast<char *>(&count), sizeof(count))) {
                    if (input.eof() && input.gcount() == 0) {
                        break;
                    }
                    throw std::runtime_error("incomplete embedding message header");
                }
            }
            wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wait_started).count();
            ++wait_calls;
            if (getenv("M5_STAGE_DEBUG"))
                std::fprintf(stderr, "m5-stage msg count=%u start=%d\n", count, start);
            if constexpr (!kRing) {
                if (count == (uint32_t) -1) {
                    // Prefix-cached prefill: [shared_rows][hash][rest_rows] then
                    // shared rows and rest rows. The shared block's KV is kept
                    // across requests when the hash matches, so only the new
                    // text plus reference codes get decoded again.
                    int32_t sh = 0, rs = 0;
                    uint32_t hash = 0;
                    if (!input.read(reinterpret_cast<char *>(&sh), sizeof(sh)) ||
                        !input.read(reinterpret_cast<char *>(&hash), sizeof(hash)) ||
                        !input.read(reinterpret_cast<char *>(&rs), sizeof(rs))) {
                        throw std::runtime_error("incomplete prefill header");
                    }
                    if (sh < 0 || rs <= 0 || (size_t) (sh + rs) > (size_t) prefix) {
                        throw std::runtime_error("prefill exceeds context");
                    }
                    llama_memory_t mem = llama_get_memory(context.get());
                    const bool hit = sh > 0 && sh == cached_shared && hash == cached_hash;
                    if (hit) {
                        // Keep [0, sh) and drop everything the last request
                        // appended; the shared block is unchanged (causal).
                        llama_memory_seq_rm(mem, sequence, sh, -1);
                    } else {
                        llama_memory_clear(mem, false);
                        cached_shared = 0;
                        cached_hash = 0;
                    }
                    auto decode_block = [&](int n, int pos) {
                        if (!input.read(reinterpret_cast<char *>(embeddings.data()),
                                        (size_t) n * row_bytes)) {
                            throw std::runtime_error("cannot read prefill rows");
                        }
                        for (int i = 0; i < n; ++i) {
                            positions[i] = pos + i;
                            requested[i] = i == n - 1;
                        }
                        llama_batch batch{};
                        batch.n_tokens = n;
                        batch.embd = embeddings.data();
                        batch.pos = positions.data();
                        batch.n_seq_id = sequence_counts.data();
                        batch.seq_id = sequence_ids.data();
                        batch.logits = requested.data();
                        if (llama_decode(context.get(), batch) != 0) {
                            throw std::runtime_error("backbone decode failed");
                        }
                    };
                    if (sh > 0) {
                        if (hit) {
                            // The engine always sends the shared block; on a hit
                            // its bytes still have to be consumed or the next
                            // message header reads garbage.
                            if (!input.read(reinterpret_cast<char *>(embeddings.data()),
                                            (size_t) sh * row_bytes)) {
                                throw std::runtime_error("cannot read prefill rows");
                            }
                        } else {
                            decode_block(sh, 0);
                            cached_shared = sh;
                            cached_hash = hash;
                        }
                    }
                    decode_block(rs, sh);
                    const auto * hidden = llama_get_embeddings_ith(context.get(), -1);
                    if (!hidden) {
                        throw std::runtime_error("cannot read backbone hidden");
                    }
                    if (!output.write(reinterpret_cast<const char *>(hidden), row_bytes).flush()) {
                        throw std::runtime_error("cannot write prefill reply");
                    }
                    start = sh + rs;
                    continue;
                }
            }
            if (count == 0) {
                if (depth && start != 0) {
                    std::fprintf(stderr, "m5-depth frame=%d core_ms=%.3f\n", frames++, prefill_ms + decode_ms);
                }
                llama_memory_clear(llama_get_memory(context.get()), false);
                prefill_ms = 0;
                decode_ms = 0;
                decode_calls = 0;
                start = 0;
                if constexpr (kRing) ring.publish();
                continue;
            }
            if (count == 0 || count > prefix || count > static_cast<uint32_t>(rows - start)) {
                throw std::runtime_error("embedding message exceeds remaining context");
            }
            if constexpr (kRing) {
                std::memcpy(embeddings.data(), message, count * row_bytes);
            } else if (!input.read(reinterpret_cast<char *>(embeddings.data()), count * row_bytes)) {
                throw std::runtime_error("cannot read embeddings");
            }
            for (uint32_t i = 0; i < count; ++i) {
                positions[i] = start + i;
                requested[i] = i == count - 1;
            }
            llama_batch batch{};
            batch.n_tokens = count;
            batch.embd = embeddings.data();
            batch.pos = positions.data();
            batch.n_seq_id = sequence_counts.data();
            batch.seq_id = sequence_ids.data();
            batch.logits = requested.data();
            const auto started = std::chrono::steady_clock::now();
            if (llama_decode(context.get(), batch) != 0) {
                throw std::runtime_error("backbone decode failed");
            }
#ifdef M5_STAGE_TIMING
            const auto decoded = std::chrono::steady_clock::now();
#endif
            const auto * hidden = llama_get_embeddings_ith(context.get(), -1);
            const auto finished = std::chrono::steady_clock::now();
#ifdef M5_STAGE_TIMING
            decode_api_ms += std::chrono::duration<double, std::milli>(decoded - started).count();
            embedding_api_ms += std::chrono::duration<double, std::milli>(finished - decoded).count();
            if (++profiled_calls % 100 == 0) {
                std::fprintf(stderr, "m5-stage timing calls=%d decode_api_ms=%.3f embedding_api_ms=%.3f\n",
                             profiled_calls, decode_api_ms, embedding_api_ms);
            }
#endif
            const auto milliseconds = std::chrono::duration<double, std::milli>(
                finished - started).count();
            if (start == 0) {
                prefill_ms = milliseconds;
            } else {
                decode_ms += milliseconds;
                ++decode_calls;
            }
            if (!hidden) {
                throw std::runtime_error("cannot write final hidden state");
            }
            if constexpr (kRing) {
                std::memcpy(ring.reply(), hidden, row_bytes);
                ring.publish();
            } else if (!output.write(reinterpret_cast<const char *>(hidden), row_bytes).flush()) {
                throw std::runtime_error("cannot write final hidden state");
            }
            start += count;
        }
        };

#if M5_STAGE_RING
        if (use_ring) {
            serve(std::true_type{});
        } else {
            serve(std::false_type{});
        }
#else
        if (use_ring) {
            // Builds without the ring reject the fd so the engine falls back to
            // pipes instead of hanging on a block this process never maps.
            std::fprintf(stderr, "this build has no shared-ring transport\n");
            return 1;
        }
        serve(std::false_type{});
#endif
        if (!pipe) {
            output_file.close();
        }
        if (!use_ring && !output) {
            throw std::runtime_error("cannot close output");
        }
        std::fprintf(stderr, "m5-backbone prefill_ms=%.3f decode_ms=%.3f decode_calls=%d wait_ms=%.3f wait_calls=%d\n",
                     prefill_ms, decode_ms, decode_calls, wait_ms, wait_calls);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
