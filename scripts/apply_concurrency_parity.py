#!/usr/bin/env python3
# Applies the vLLM-concurrency-parity integration edits across all touched
# files. Idempotent: each edit first checks whether its anchor is already
# present, and skips if so. Run repeatedly until it reports all edits applied.
#
# The concurrent server-rewrite agent keeps overwriting the shared headers and
# server-context.cpp; this script re-applies the integration deterministically.
import sys

def load(p):
    with open(p) as f:
        return f.read()

def save(p, s):
    with open(p, "w") as f:
        f.write(s)

def replace_once(src, anchor_present, old, new):
    if anchor_present in src:
        return src, "skip(present)"
    if old not in src:
        return src, "MISSING-OLD"
    return src.replace(old, new, 1), "applied"

def insert_after_once(src, anchor_present, marker, new_block):
    if anchor_present in src:
        return src, "skip(present)"
    idx = src.find(marker)
    if idx < 0:
        return src, "MISSING-MARKER"
    end = idx + len(marker)
    return src[:end] + new_block + src[end:], "applied"

results = []

# ===========================================================================
# common/common.h - add params fields after kv_unified
# ===========================================================================
src = load("common/common.h")
ANCHOR = "    bool kv_unified        = true; // unified KV cache: single stream shared across sequences"
ADD = """

    // --- vLLM-concurrency parity options (see docs/vllm-concurrency-study.md) ---
    // Chunked prefill: shared per-iteration prefill cap (Sarathi-Serve).
    // 0 = disabled (legacy). Default matches vLLM V1's long_prefill_token_threshold.
    int32_t prefill_chunk_size   = 8192;
    // Dynamic admission + recompute preemption (vLLM V1 policy).
    // 0 = controller disabled (legacy). When the number of in-flight requests
    // reaches this cap, the lowest-scoring active request is preempted.
    int32_t max_admitted         = 256;
    // Cap preemptions per iteration to avoid thundering-herd thrash.
    int32_t max_preemptions_per_iter = 1;

    // --- Observability (latency histograms + OTel-style tracing) ---
    bool        otel_enabled         = false;
    std::string otel_endpoint;            // empty = stderr ndjson, else OTLP/HTTP URL
    std::string otel_service_name  = "tessera-server";
    double      otel_sample_rate   = 1.0; // [0,1] sampling probability"""
# Try the current agent spelling first, fall back to upstream default
for old_line in [
    "    bool kv_unified        = true; // unified KV cache: single stream shared across sequences",
    "    bool kv_unified        = false; // enable unified KV cache",
]:
    src, st = replace_once(src, "prefill_chunk_size", old_line, old_line + ADD)
    if st == "applied":
        break
save("common/common.h", src)
results.append(("common.h params", st))

# ===========================================================================
# common/arg.cpp - add CLI flags after cont-batching
# ===========================================================================
src = load("common/arg.cpp")
CB_LINE = '    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));'
FLAGS = '''
    add_opt(common_arg(
        {"--prefill-chunk-size"}, "N",
        string_format(
            "shared per-iteration prefill token budget (chunked prefill / Sarathi-Serve). "
            "Caps the total prefill tokens processed in one iteration across all slots, so a "
            "single long prompt cannot starve running decodes. 0 disables the cap. "
            "(default: %d, matching vLLM V1)",
            params.prefill_chunk_size),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for prefill_chunk_size\\n");
            }
            params.prefill_chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PREFILL_CHUNK_SIZE"));
    add_opt(common_arg(
        {"--max-admitted"}, "N",
        string_format(
            "soft cap on in-flight requests before recompute preemption kicks in "
            "(vLLM V1 dynamic admission). When the number of processing requests reaches this "
            "cap, the lowest-scoring active request is preempted (its KV discarded, task "
            "re-queued) to admit a higher-scoring arrival. 0 disables the controller. "
            "Set lower than n_parallel to allow preemption. (default: %d)",
            params.max_admitted),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for max_admitted\\n");
            }
            params.max_admitted = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MAX_ADMITTED"));
    add_opt(common_arg(
        {"--otel-endpoint"}, "URL",
        "enable OpenTelemetry-compatible tracing and export spans to this OTLP/HTTP endpoint "
        "(e.g. http://collector:4318/v1/traces). When omitted, tracing is off. "
        "Implementation is a lightweight W3C trace-context + newline-delimited JSON exporter "
        "(see server-metrics.h); the full OTel C++ SDK is intentionally not pulled in to keep "
        "the single-binary portability guarantee.",
        [](common_params & params, const std::string & value) {
            params.otel_enabled  = true;
            params.otel_endpoint = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_OTEL_ENDPOINT"));
    add_opt(common_arg(
        {"--otel-service-name"}, "NAME",
        string_format(
            "service.name attribute for OTel spans (default: '%s')",
            params.otel_service_name.c_str()),
        [](common_params & params, const std::string & value) {
            params.otel_service_name = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_OTEL_SERVICE_NAME"));
    add_opt(common_arg(
        {"--otel-sample-rate"}, "R",
        string_format(
            "OTel span sampling probability in [0,1]. 1.0 records every request, 0.05 keeps "
            "tracing cheap on a busy server. (default: %.2f)",
            params.otel_sample_rate),
        [](common_params & params, const std::string & value) {
            double r = std::stod(value);
            if (r < 0.0 || r > 1.0) {
                throw std::invalid_argument("error: --otel-sample-rate must be in [0,1]\\n");
            }
            params.otel_sample_rate = r;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_OTEL_SAMPLE_RATE"));'''
# Only the first cont-batching line gets the flags appended.
idx = src.find(CB_LINE)
if "--prefill-chunk-size" in src:
    st = "skip(present)"
elif idx < 0:
    st = "MISSING-MARKER"
else:
    src = src[:idx + len(CB_LINE)] + FLAGS + src[idx + len(CB_LINE):]
    st = "applied"
save("common/arg.cpp", src)
results.append(("arg.cpp flags", st))

# ===========================================================================
# tools/server/server-task.h - add preempted_from + result fields
# ===========================================================================
src = load("tools/server/server-task.h")
src, st = replace_once(
    src,
    "int preempted_from = -1;",
    "    task_params   params;\n    server_tokens tokens;\n",
    "    task_params   params;\n    server_tokens tokens;\n\n    // diagnostic: id of the original task this request was preempted from, or\n    // -1 if it was never preempted. Set by the admission controller when it\n    // re-queues a preempted request so the operator can correlate the retry\n    // with the victim in logs / metrics.\n    int preempted_from = -1;\n")
results.append(("server-task.h preempted_from", st))

src, st = replace_once(
    src,
    "uint64_t n_prefill_shared_cap_yields_total = 0;",
    "    int64_t  scheduler_score_milli_last = 0;\n\n    // while we can also use std::vector<server_slot>",
    "    int64_t  scheduler_score_milli_last = 0;\n\n    // Change 1: shared per-iteration prefill cap yields.\n    uint64_t n_prefill_shared_cap_yields_total = 0;\n    // Change 2: dynamic admission + recompute preemption.\n    uint64_t n_admit_attempts_total    = 0;\n    uint64_t n_admit_preemptions_total = 0;\n\n    // Change 3: pre-rendered Prometheus histogram exposition text (TTFT, ITL,\n    // e2e, prefill, decode). Carried as a string because the latency_registry\n    // owns the bucket layout and already renders the Prometheus format.\n    std::string latency_histograms_prometheus;\n\n    // while we can also use std::vector<server_slot>")
save("tools/server/server-task.h", src)
results.append(("server-task.h result fields", st))

# ===========================================================================
# tools/server/server-context.cpp - the full integration
# ===========================================================================
PATH = "tools/server/server-context.cpp"
src = load(PATH)

edits = []

src, st = replace_once(
    src, '#include "server-metrics.h"',
    '#include "server-stream.h"\n\n#include "build-info.h"',
    '#include "server-stream.h"\n#include "server-metrics.h"\n#include "server-prefill-policy.h"\n#include "server-admission.h"\n\n#include "build-info.h"')
edits.append(("includes", st))

MEM_NEW = '''

    // --- vLLM-concurrency parity modules (see docs/vllm-concurrency-study.md) ---
    // Latency histograms + OTel-style tracing (Change 3).
    tessera_metrics::registry        latency_registry;
    tessera_metrics::tracer          tracer;
    // Shared per-iteration prefill cap (Change 1). Reused across iterations;
    // reset() is called at the top of each pre_decode().
    tessera_prefill::budget_calculator prefill_budget{tessera_prefill::policy_config{}};
    uint64_t n_prefill_shared_cap_yields_total = 0;
    // Dynamic admission + recompute preemption bookkeeping (Change 2).
    uint64_t n_admit_preemptions_total     = 0;
    uint64_t n_admit_attempts_total        = 0;
    int32_t  preemptions_this_iter         = 0;
'''
src, st = replace_once(
    src, "tessera_metrics::registry        latency_registry;",
    "    server_metrics metrics;\n\n    json json_ui_settings = json::object();",
    "    server_metrics metrics;" + MEM_NEW + "\n    json json_ui_settings = json::object();")
edits.append(("members", st))

CFG_NEW = '''

        // Configure the vLLM-concurrency-parity modules from CLI flags.
        tessera_prefill::policy_config pcfg;
        pcfg.iteration_cap_tokens = params_base.prefill_chunk_size;
        prefill_budget.configure(pcfg);

        tessera_metrics::tracer::config tcfg;
        tcfg.enabled      = params_base.otel_enabled;
        tcfg.endpoint     = params_base.otel_endpoint;
        tcfg.service_name = params_base.otel_service_name;
        tcfg.sample_rate  = params_base.otel_sample_rate;
        tracer.reconfigure(tcfg);
'''
src, st = replace_once(
    src, "tessera_prefill::policy_config pcfg;",
    "        params_base.n_outputs_max = server_n_outputs_max(params_base);\n\n        const bool has_embedded_mtp",
    "        params_base.n_outputs_max = server_n_outputs_max(params_base);" + CFG_NEW + "\n        const bool has_embedded_mtp")
edits.append(("load_model cfg", st))

HELPERS = '''

    // Change 3: record the end-of-request latency distributions. Called from
    // every request completion path so e2e / decode histograms see every
    // finished request exactly once. No-op when slot timings are unset
    // (e.g. an empty-prompt early release).
    void record_completion_latency(const server_slot & slot) {
        if (slot.t_start_process_prompt <= 0) {
            return;
        }
        const int64_t now_us = ggml_time_us();
        latency_registry.record_e2e(now_us - slot.t_start_process_prompt);
        if (slot.t_token_generation > 0.0) {
            // decode histogram tracks per-request decode wall time so it is
            // comparable to vLLM's request_decode_time_seconds.
            latency_registry.record_decode(
                (int64_t)(slot.t_token_generation * 1000.0));
        }
    }

    // Change 2: dynamic admission + recompute preemption.
    //
    // Build a snapshot of every slot (with its current scheduler score) and
    // ask the admission controller whether the incoming candidate should
    // displace a running request. If yes, the victim's KV is discarded
    // (recompute preemption, vLLM V1 policy), its slot is released back to
    // the idle pool, and its original task is re-queued. Returns the freed
    // slot pointer so the caller can immediately admit the candidate, or
    // nullptr if no preemption was performed.
    server_slot * maybe_preempt_for(int64_t candidate_score) {
        if (params_base.max_admitted <= 0) {
            return nullptr;
        }
        if (preemptions_this_iter >= params_base.max_preemptions_per_iter) {
            return nullptr;
        }

        tessera_admission::controller_snapshot snap;
        snap.n_processing = 0;
        const int64_t now_us = ggml_time_us();
        snap.slots.reserve(slots.size());
        for (const server_slot & s : slots) {
            tessera_admission::slot_snapshot ss;
            ss.id          = s.id;
            ss.processing  = s.is_processing();
            if (ss.processing) {
                ++snap.n_processing;
            }
            ss.score_milli = score_prefill_slot(s, now_us).score_milli;
            ss.remaining_prefill = (int32_t) std::max<int>(
                0, (s.task ? s.task->n_tokens() : 0) - s.prompt.n_tokens());
            ss.decoded_so_far    = s.n_decoded;
            // Parent slots driving parallel-sampling children must not be
            // preempted: doing so would orphan the children. Child slots are
            // fair game.
            ss.preemptable = s.is_processing() && s.task &&
                             !s.task->is_parent();
            snap.slots.push_back(std::move(ss));
        }

        tessera_admission::admission_config acfg;
        acfg.max_admitted              = params_base.max_admitted;
        acfg.max_preemptions_per_iter  = params_base.max_preemptions_per_iter;
        acfg.min_score_advantage_milli = 0;

        auto victim_id = tessera_admission::choose_preempt_victim(
            snap, candidate_score, acfg);
        if (!victim_id.has_value()) {
            return nullptr;
        }

        server_slot * victim = get_slot_by_id(*victim_id);
        if (victim == nullptr || !victim->is_processing() || !victim->task) {
            return nullptr;
        }

        // Snapshot what we need to rebuild the task, then release the victim.
        // server_tokens is move-only, so we move the victim's tokens out
        // before release() clears the task. params is copyable.
        const int     victim_task_id  = victim->task->id;
        const bool    victim_is_cmpl  = victim->task->type == SERVER_TASK_TYPE_COMPLETION;
        server_tokens victim_tokens;
        task_params   victim_params;
        if (victim_is_cmpl) {
            victim_tokens = std::move(victim->task->tokens);
            victim_params = victim->task->params;
        }

        SLT_INF(*victim, "preempting for recompute admission (victim score=%" PRId64
                ", candidate score=%" PRId64 ", n_processing=%d)\\n",
                snap.slots[*victim_id].score_milli, candidate_score, snap.n_processing);

        // Discard the victim's KV (the "recompute" half of recompute
        // preemption). release() moves the task into task_prev and frees the
        // slot; prompt_clear() then drops the KV cells.
        victim->release();
        victim->prompt_clear();
        if (params_base.kv_unified) {
            kv_block_radix.release(victim->id);
        }
        ++n_admit_preemptions_total;
        ++preemptions_this_iter;

        // Re-queue the preempted task so it gets re-admitted (and its prefix
        // re-attached from the radix cache) when a slot opens up.
        if (victim_is_cmpl) {
            server_task retry(SERVER_TASK_TYPE_COMPLETION);
            retry.id         = queue_tasks.get_new_id();
            retry.index      = 0;
            retry.params     = std::move(victim_params);
            retry.tokens     = std::move(victim_tokens);
            retry.preempted_from = victim_task_id;  // diagnostic link
            SLT_INF(*victim, "re-queued preempted task, new id=%d\\n", retry.id);
            queue_tasks.post(std::move(retry));
        }

        return victim;
    }
'''
src, st = insert_after_once(
    src, "void record_completion_latency(const server_slot & slot)",
    "        kv_block_radix.publish(keys, positions, slot.id, ggml_time_us());\n    }",
    HELPERS)
edits.append(("helpers", st))

RESET_NEW = '''
            // Reset the shared per-iteration prefill budget (Change 1).
            // already_in_batch counts the decodes already staged this iteration;
            // n_decodes_pending reserves room for the rest of the generating
            // slots that have not yet emitted their decode token. When the cap
            // is hit, request_quantum() returns a smaller quantum for the
            // remaining prefill slots so decode is never fully starved.
            prefill_budget.reset(n_batch,
                                 batch.size(),
                                 (int32_t) generating.size());

'''
src, st = replace_once(
    src, "prefill_budget.reset(n_batch,",
    "            }\n\n            const int64_t schedule_now_us = ggml_time_us();\n            std::stable_sort(prefill_slots.begin(),",
    "            }\n" + RESET_NEW + "            const int64_t schedule_now_us = ggml_time_us();\n            std::stable_sort(prefill_slots.begin(),")
edits.append(("prefill reset", st))

QUANTUM_OLD = """                    const int32_t prefill_quantum = prefill_quantum_for(
                        slot, n_batch, n_ubatch, !generating.empty());
                    metrics.on_prefill_adaptive_quantum(
                        prefill_quantum, prefill_quantum_max);"""
QUANTUM_NEW = """                    const int32_t forced_quantum = prefill_quantum_for(
                        slot, n_batch, n_ubatch, !generating.empty());
                    metrics.on_prefill_adaptive_quantum(
                        forced_quantum, prefill_quantum_max);
                    // Apply the shared per-iteration prefill cap (Change 1).
                    // The cap may reduce this slot's contribution below
                    // forced_quantum to leave room for other prefills or for
                    // running decodes in the same iteration.
                    const int32_t slot_remaining = (int32_t) std::max<int>(
                        0, slot.task->n_tokens() - slot.prompt.n_tokens());
                    const int32_t prefill_quantum = prefill_budget.request_quantum(
                        slot_remaining, forced_quantum);"""
src, st = replace_once(src, "const int32_t forced_quantum = prefill_quantum_for(", QUANTUM_OLD, QUANTUM_NEW)
edits.append(("prefill quantum", st))

YIELD_OLD = """                    if (!generating.empty() &&
                        n_tokens_cur >= prefill_quantum &&
                        slot.prompt.n_tokens() < slot.task->n_tokens()) {
                        metrics.on_prefill_quantum_yield();
                    }
"""
YIELD_NEW = """                    if (!generating.empty() &&
                        n_tokens_cur >= prefill_quantum &&
                        slot.prompt.n_tokens() < slot.task->n_tokens()) {
                        metrics.on_prefill_quantum_yield();
                    }
                    // Record a shared-cap yield separately: the slot had more
                    // prompt to ingest but the per-iteration budget reduced or
                    // zeroed its quantum this step.
                    if (prefill_quantum < forced_quantum &&
                        slot.prompt.n_tokens() < slot.task->n_tokens()) {
                        ++n_prefill_shared_cap_yields_total;
                    }
"""
src, st = replace_once(src, "++n_prefill_shared_cap_yields_total;", YIELD_OLD, YIELD_NEW)
edits.append(("shared-cap yield", st))

DEC_OLD = """        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
"""
DEC_NEW = """        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            // Capture the decode wall time for the per-iteration latency
            // histogram (Change 3). Measured here rather than via scoped_timer
            // so it works in both DEBUG_TIMINGS and release builds.
            const int64_t t_decode_iter_start = ggml_time_us();
            try {
                scoped_timer t(t_decode, n_decode);
"""
src, st = replace_once(src, "const int64_t t_decode_iter_start = ggml_time_us();", DEC_OLD, DEC_NEW)
edits.append(("decode timer start", st))

DEC2_OLD = """                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }"""
DEC2_NEW = """                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);

                    // Change 3: per-iteration decode latency distribution.
                    latency_registry.record_decode(
                        (int64_t)(ggml_time_us() - t_decode_iter_start));
                } else {
                    // try again with the updated n_batch
                    continue;
                }"""
src, st = replace_once(src, "latency_registry.record_decode(\n                        (int64_t)(ggml_time_us() - t_decode_iter_start));", DEC2_OLD, DEC2_NEW)
edits.append(("decode record", st))

RESET_ITER_OLD = """#endif

        // check if all slots are idle
        {
            bool all_idle = true;"""
RESET_ITER_NEW = """#endif

        // Reset the per-iteration preemption counter (Change 2). The admission
        // controller respects max_preemptions_per_iter to avoid thundering-
        // herd thrash when many large prompts arrive together.
        preemptions_this_iter = 0;

        // check if all slots are idle
        {
            bool all_idle = true;"""
src, st = replace_once(src, "preemptions_this_iter = 0;", RESET_ITER_OLD, RESET_ITER_NEW)
edits.append(("preempt reset", st))

FIRST_OLD = """            if (slot.n_decoded == 1) {
                slot.t_start_generation = t_now;
                slot.t_print_last = t_now;
                slot.n_decoded_last = 0;
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
            }

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;"""
FIRST_NEW = """            if (slot.n_decoded == 1) {
                slot.t_start_generation = t_now;
                slot.t_print_last = t_now;
                slot.n_decoded_last = 0;
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
                // Change 3: TTFT = arrival -> first generated token. Also
                // record the prefill latency distribution.
                if (slot.t_start_process_prompt > 0) {
                    latency_registry.record_ttft(
                        (int64_t)((slot.t_start_generation - slot.t_start_process_prompt)));
                    latency_registry.record_prefill(
                        (int64_t)(slot.t_prompt_processing * 1000.0));
                }
            } else {
                // Change 3: inter-token latency, measured between the
                // previous accepted token (t_print_last) and this one.
                if (slot.t_print_last > 0) {
                    latency_registry.record_itl((int64_t)(t_now - slot.t_print_last));
                }
            }
            slot.t_print_last = t_now;

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;"""
src, st = replace_once(src, "latency_registry.record_ttft(", FIRST_OLD, FIRST_NEW)
edits.append(("ttft/itl", st))

def add_completion_latency(src, indent_spaces):
    if "record_completion_latency(slot);" in src:
        return src, "skip(present)"
    old = indent_spaces + "metrics.on_prediction(slot);\n" + indent_spaces + "slot.release();\n"
    new = indent_spaces + "metrics.on_prediction(slot);\n" + indent_spaces + "record_completion_latency(slot);\n" + indent_spaces + "slot.release();\n"
    if old not in src:
        return src, "MISSING-OLD"
    return src.replace(old, new, 1), "applied"
src, st = add_completion_latency(src, "                ")
edits.append(("completion-latency #1", st))
src, st = add_completion_latency(src, "                    ")
edits.append(("completion-latency #2", st))

NOSLOT_OLD = """                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }"""
NOSLOT_NEW = """                    if (slot == nullptr) {
                        // No idle slot. Try dynamic admission: if the
                        // admission controller is enabled and this candidate
                        // outranks a running request, preempt the victim by
                        // discarding its KV (recompute) and reuse its slot
                        // (Change 2). Otherwise defer as usual.
                        ++n_admit_attempts_total;
                        if (task.type == SERVER_TASK_TYPE_COMPLETION) {
                            // Estimate the candidate's scheduler score from its
                            // radix-cache prefix hit (the dominant term), with
                            // age = 0. A real slot does not exist yet.
                            llama_pos cand_prefix = 0;
                            if (task.params.cache_prompt && prompt_cache) {
                                if (auto * cached = prompt_cache->find_longest_prefix(task.tokens)) {
                                    cand_prefix = cached->prompt.tokens.pos_next();
                                }
                            }
                            const int64_t candidate_score =
                                  1000ll * cand_prefix
                                + 16ll * 500; // neutral acceptance_milli
                            server_slot * freed = maybe_preempt_for(candidate_score);
                            if (freed != nullptr) {
                                slot = freed;
                            }
                        }
                        if (slot == nullptr) {
                            // if no slot is available, we defer this task for processing later
                            SRV_DBG("no slot is available, defer task, id_task = %d\\n", id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                    }"""
src, st = replace_once(src, "server_slot * freed = maybe_preempt_for(candidate_score);", NOSLOT_OLD, NOSLOT_NEW)
edits.append(("no-slot preempt", st))

MH_OLD = """                    res->scheduler_score_milli_last = metrics.scheduler_score_milli_last;

                    if (task.metrics_reset_bucket) {"""
MH_NEW = """                    res->scheduler_score_milli_last = metrics.scheduler_score_milli_last;

                    // Change 1 / 2 / 3 counters + histograms.
                    res->n_prefill_shared_cap_yields_total = n_prefill_shared_cap_yields_total;
                    res->n_admit_attempts_total    = n_admit_attempts_total;
                    res->n_admit_preemptions_total = n_admit_preemptions_total;
                    latency_registry.render_prometheus(res->latency_histograms_prometheus);

                    if (task.metrics_reset_bucket) {"""
src, st = replace_once(src, "res->n_prefill_shared_cap_yields_total = n_prefill_shared_cap_yields_total;", MH_OLD, MH_NEW)
edits.append(("metrics handler copy", st))

# Counters in the counter array
CTR_OLD = """            }, {
                    {"name",  "scheduler_acceptance_milli_total"},
                    {"help",  "Rolling speculative acceptance contribution, scaled by 1000."},
                    {"value",  res_task->scheduler_acceptance_milli_total}
            }, {
                    {"name",  "n_tokens_max"},"""
CTR_NEW = """            }, {
                    {"name",  "scheduler_acceptance_milli_total"},
                    {"help",  "Rolling speculative acceptance contribution, scaled by 1000."},
                    {"value",  res_task->scheduler_acceptance_milli_total}
            }, {
                    {"name",  "prefill_shared_cap_yields_total"},
                    {"help",  "Prompt chunks reduced or deferred by the shared per-iteration prefill cap (chunked prefill, Change 1)."},
                    {"value",  res_task->n_prefill_shared_cap_yields_total}
            }, {
                    {"name",  "admit_attempts_total"},
                    {"help",  "Task arrivals that found no idle slot and consulted the dynamic admission controller (Change 2)."},
                    {"value",  res_task->n_admit_attempts_total}
            }, {
                    {"name",  "admit_preemptions_total"},
                    {"help",  "Active requests preempted via KV discard (recompute) to admit a higher-scoring arrival (Change 2)."},
                    {"value",  res_task->n_admit_preemptions_total}
            }, {
                    {"name",  "n_tokens_max"},"""
src, st = replace_once(src, "prefill_shared_cap_yields_total\",\n                    {\"help\",  \"Prompt chunks reduced", CTR_OLD, CTR_NEW)
edits.append(("counter array", st))

# Histogram append to prometheus render
HIST_OLD = """                prometheus << "# HELP llamacpp:" << name << " " << help  << "\\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\\n"
                            << "llamacpp:"        << name << " " << value << "\\n";
            }
        }

        res->headers["Process-Start-Time-Unix"]"""
HIST_NEW = """                prometheus << "# HELP llamacpp:" << name << " " << help  << "\\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\\n"
                            << "llamacpp:"        << name << " " << value << "\\n";
            }
        }

        // Change 3: append the pre-rendered latency histograms (TTFT, ITL,
        // e2e, prefill, decode). Each block is a full Prometheus histogram
        // family with buckets, _sum, _count, plus P50/P95/P99 gauges.
        if (!res_task->latency_histograms_prometheus.empty()) {
            prometheus << res_task->latency_histograms_prometheus;
        }

        res->headers["Process-Start-Time-Unix"]"""
src, st = replace_once(src, "if (!res_task->latency_histograms_prometheus.empty()) {", HIST_OLD, HIST_NEW)
edits.append(("histogram append", st))

save(PATH, src)
for name, st in edits:
    results.append((name, st))

print("Edits:")
for name, st in results:
    print(f"  {st:18s} {name}")
n_missing = sum(1 for _, st in results if st.startswith("MISSING"))
sys.exit(0 if n_missing == 0 else 1)
