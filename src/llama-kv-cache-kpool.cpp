#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cmath>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    return n_kv/kpool + 2*n_seqs;
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_pools, indexer_top_k/kpool);
}

// sel_mask and cand_mask hold only 0.0f and -INFINITY, so f16 is exact here
template <typename T> struct kpool_mask_of;

template <> struct kpool_mask_of<float> {
    static float from(float v) { return v; }
};

template <> struct kpool_mask_of<ggml_fp16_t> {
    static ggml_fp16_t from(float v) { return ggml_fp32_to_fp16(v); }
};

template <typename T>
static void kpool_mask_fill(T * dst, int64_t n) {
    std::fill(dst, dst + n, kpool_mask_of<T>::from(-INFINITY));
}

template <typename T>
static void kpool_mask_row(
                T * cur_sel,
                T * cur_cand,
        const llama_pos * pos_at,
        const int32_t   * pool_of,
          int64_t   n_kv,
        llama_pos   q,
        llama_pos   tail_start,
          int64_t   bo_vis) {
    const T v_sel  = kpool_mask_of<T>::from(0.0f);
    const T v_mask = kpool_mask_of<T>::from(-INFINITY);

    for (int64_t j = 0; j < n_kv; ++j) {
        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;
        const bool tail   = pos_at[j] >= tail_start;

        cur_sel [j] = vis && tail   ? v_sel : v_mask;
        cur_cand[j] = vis && (pooled || tail) ? v_sel : v_mask;
    }
}

void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        const uint32_t       * strm_of,
              int64_t          kv_size,
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool) {
    GGML_ASSERT(kv != nullptr);
    GGML_ASSERT(kpool > 0);

    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_bias ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask  ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask ->buffer));

    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias ->type == GGML_TYPE_F32);
    GGML_ASSERT((sel_mask->type == GGML_TYPE_F16 || sel_mask->type == GGML_TYPE_F32) &&
            "sel_mask must be f16 or f32");
    GGML_ASSERT(cand_mask->type == sel_mask->type && "both masks must have the KQ mask's type");

    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(pool_bias));
    GGML_ASSERT(ggml_is_contiguous(sel_mask));
    GGML_ASSERT(ggml_is_contiguous(cand_mask));

    const int64_t n_kv     = sel_mask->ne[0];
    const int64_t n_ns     = sel_mask->ne[3];
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION] positions are unambiguous only within one sequence, so one
    // pool map per SEQUENCE, not per stream
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;
    const int64_t n_pools = pool_cells->ne[0]/r;

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(sel_mask->ne[2] == 1);
    GGML_ASSERT(ggml_are_same_shape(cand_mask, sel_mask));
    GGML_ASSERT(pool_bias->ne[0] == n_pools && pool_bias->ne[2] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps  = n_tokens/n_ns;
    const int64_t n_padq = sel_mask->ne[1];

    GGML_ASSERT(pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);

    if (cell_pool) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
        GGML_ASSERT(cell_pool->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(cell_pool));
        GGML_ASSERT(cell_pool->ne[0] == n_kv && cell_pool->ne[1] == n_ns);

        // one row per stream, so a shared cell has nowhere to put its second pool
        GGML_ASSERT(n_ps == 1 && "the per-cell pool view needs one sequence per stream");
    }

    if (bias) {
        GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[1] == n_tps && bias->ne[2] == n_ns);
    }

    const bool kcache = pool_reps != nullptr;

    GGML_ASSERT((new_pool_cells != nullptr) == kcache);
    GGML_ASSERT((new_pool_reps  != nullptr) == kcache);

    int64_t n_new_max = 0;

    if (kcache) {
        GGML_ASSERT(ggml_backend_buffer_is_host(pool_reps     ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_reps ->buffer));

        GGML_ASSERT(pool_reps     ->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_reps ->type == GGML_TYPE_I64);

        GGML_ASSERT(ggml_is_contiguous(pool_reps));
        GGML_ASSERT(ggml_is_contiguous(new_pool_cells));
        GGML_ASSERT(ggml_is_contiguous(new_pool_reps));

        GGML_ASSERT(pool_reps->ne[0] == n_pools && pool_reps->ne[1] == n_ns);
        GGML_ASSERT(new_pool_cells->ne[0] % r == 0 && new_pool_cells->ne[1] == n_ns);

        n_new_max = new_pool_cells->ne[0]/r;

        GGML_ASSERT(new_pool_reps->ne[0] == n_new_max*n_ns);
        GGML_ASSERT(strm_of != nullptr && kv_size > 0);
    }

    int32_t * dst_pool_reps = kcache ? (int32_t *) pool_reps     ->data : nullptr;
    int32_t * dst_new_cells = kcache ? (int32_t *) new_pool_cells->data : nullptr;
    int64_t * dst_new_reps  = kcache ? (int64_t *) new_pool_reps ->data : nullptr;

    int32_t * dst_cell_pool  = cell_pool ? (int32_t *) cell_pool->data : nullptr;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = bias ? (float *) bias->data : nullptr;
    float   * dst_pool_bias  = (float   *) pool_bias ->data;
    char    * dst_sel_mask   = (char    *) sel_mask  ->data;
    char    * dst_cand_mask  = (char    *) cand_mask ->data;

    const bool   mask_f16 = sel_mask->type == GGML_TYPE_F16;
    const size_t mask_ts  = ggml_type_size(sel_mask->type);

    // -1 marks a cell with no usable pool; host side only, never copied into cell_pool
    std::vector<int32_t>   pool_of(n_kv);
    std::vector<int32_t>   filled(n_pools);
    std::vector<llama_pos> pos_at;

    std::vector<int64_t> run_off(n_ps);
    std::vector<int64_t> run_len(n_ps);

    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pools);
        char    * cur_sel_mask   = dst_sel_mask   + s*(n_padq*n_kv)*mask_ts;
        char    * cur_cand_mask  = dst_cand_mask  + s*(n_padq*n_kv)*mask_ts;
        float   * cur_pool_bias  = dst_pool_bias  + s*(n_tps*n_pools);

        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);
        std::fill(cur_pool_bias,  cur_pool_bias  + n_tps*n_pools, -INFINITY);

        int32_t * cur_pool_reps = kcache ? dst_pool_reps + s*n_pools        : nullptr;
        int32_t * cur_new_cells = kcache ? dst_new_cells + s*(r*n_new_max)  : nullptr;
        int64_t * cur_new_reps  = kcache ? dst_new_reps  + s*n_new_max      : nullptr;

        int64_t n_new = 0;

        // pads the fixed-size write; recomputing a complete pool is idempotent, so a repeat is safe
        const int32_t * any_rep_src = nullptr;

        if (kcache) {
            // a pool with no rep gathers row 0; such a pool is -INFINITY in pool_bias, so discarded
            std::fill(cur_pool_reps, cur_pool_reps + n_pools, 0);
            std::fill(cur_new_cells, cur_new_cells + r*n_new_max, 0);
        }

        if (mask_f16) {
            kpool_mask_fill((ggml_fp16_t *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            kpool_mask_fill((ggml_fp16_t *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
        } else {
            kpool_mask_fill((float *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            kpool_mask_fill((float *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
        }

        // [TAG_KPOOL_PACK] one packed run per sequence, NOT one full-width table: the indexer
        // scores every slot against every query, which would multiply the score tensor by n_seq_max
        {
            int64_t n_want = 0;

            for (int64_t ps = 0; ps < n_ps; ++ps) {
                const llama_seq_id seq = seq_of(s, ps);
                const auto & cells = kv->get_cells(seq);

                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                        continue;
                    }
                    const int64_t b = cells.pos_get(j)/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                run_len[ps] = found ? b_max - b_min + 1 : 0;
                n_want += run_len[ps];
            }

            if (n_want > n_pools) {
                int64_t rem = n_pools;

                for (int64_t ps = 0; ps < n_ps; ++ps) {
                    run_len[ps] = std::min(run_len[ps], rem/(n_ps - ps));
                    rem -= run_len[ps];
                }
            }

            int64_t off = 0;
            for (int64_t ps = 0; ps < n_ps; ++ps) {
                run_off[ps] = off;
                off += run_len[ps];
            }

            GGML_ASSERT(off <= n_pools);
        }

        int64_t n_done = 0;

        for (int64_t ps = 0; ps < n_ps; ++ps) {
            const llama_seq_id seq_of_pool = seq_of(s, ps);
            const auto & cells = kv->get_cells(seq_of_pool);

            const int64_t n_run = run_len[ps];

            int32_t * cur_cell_pool   = dst_cell_pool ? dst_cell_pool + s*n_kv : nullptr;
            int32_t * part_pool_cells = cur_pool_cells + run_off[ps]*r;

            std::fill(pool_of.begin(), pool_of.end(), -1);
            std::fill(filled.begin(),  filled.end(),   0);

            pos_at.resize(n_kv);
            for (int64_t j = 0; j < n_kv; ++j) {
                pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_pool) ? -1 : cells.pos_get(j);
            }

            // anchor at the absolute p/kpool (vLLM, SGLang; not HF's valid_keys.argmax(-1)): the only
            // anchor that keeps a pool's identity stable from prefill to the decodes that read it
            int64_t b_base = 0;
            {
                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (pos_at[j] < 0) {
                        continue;
                    }
                    const int64_t b = pos_at[j]/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                b_base = std::max(b_min, b_max - (n_run - 1));
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }

                const llama_pos p  = pos_at[j];
                const int64_t   bo = p/r - b_base;

                if (bo < 0 || bo >= n_run) {
                    continue;
                }

                pool_of[j] = (int32_t) bo;
                part_pool_cells[bo*r + (p%r)] = (int32_t) j;
                filled[bo]++;
            }

            // pool_valid = grouped_valid_keys.all(-1): the compressor consumes all r keys
            for (int64_t j = 0; j < n_kv; ++j) {
                // != rather than <: two cells claiming one position overwrite each other
                if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                    pool_of[j] = -1;
                }
                if (cur_cell_pool) {
                    cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
                }
            }

            if (kcache) {
                // a complete pool's key lives in the row of its LAST member; a partial pool leaves that
                // slot 0, and cell 0 is a real cell
                for (int64_t p = 0; p < n_run; ++p) {
                    if (filled[p] == (int32_t) r) {
                        cur_pool_reps[run_off[ps] + p] = part_pool_cells[p*r + (r - 1)];

                        if (any_rep_src == nullptr) {
                            any_rep_src = part_pool_cells + p*r;
                        }
                    }
                }

                // touched[] is over the run, so the cost is O(tokens), not O(n_kv)
                std::vector<uint8_t> touched(n_run, 0);

                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const int64_t i = s*n_tps + ii;

                    if (ubatch->seq_id[i][0] != seq_of_pool) {
                        continue;
                    }

                    const int64_t bo = ubatch->pos[i]/r - b_base;

                    if (bo >= 0 && bo < n_run) {
                        touched[bo] = 1;
                    }
                }

                for (int64_t p = 0; p < n_run; ++p) {
                    if ((!touched[p] && !rebuild) || filled[p] != (int32_t) r) {
                        continue;
                    }

                    // bounded while a sequence's ubatch tokens are a contiguous run (llama-batch.cpp enforces);
                    // fail loudly, clamping would serve a stale key
                    GGML_ASSERT(n_new < n_new_max && "k-pool: more pools completed than the fixed bound");

                    std::copy(part_pool_cells + p*r, part_pool_cells + (p + 1)*r,
                            cur_new_cells + n_new*r);

                    cur_new_reps[n_new] = (int64_t) strm_of[s]*kv_size + part_pool_cells[p*r + (r - 1)];

                    n_new++;
                }
            }

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int64_t   i = s*n_tps + ii;

                if (ubatch->seq_id[i][0] != seq_of_pool) {
                    continue;
                }

                const llama_pos q = ubatch->pos[i];

                GGML_ASSERT(q >= 0);

                n_done++;

                const llama_pos tail_start = (q + 1)/r*r;

                // the reference tests visibility at a pool's LAST member, so a straddled pool drops whole
                const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;
                char  * cur_sel  = cur_sel_mask  + ii*n_kv*mask_ts;
                char  * cur_cand = cur_cand_mask + ii*n_kv*mask_ts;

                if (mask_f16) {
                    kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                            pos_at.data(), pool_of.data(), n_kv, q, tail_start, bo_vis);
                } else {
                    kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                            pos_at.data(), pool_of.data(), n_kv, q, tail_start, bo_vis);
                }

                if (cur_bias) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;

                        cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                    }
                }

                float * q_pool_bias = cur_pool_bias + ii*n_pools + run_off[ps];

                for (int64_t p = 0; p < n_run; ++p) {
                    const bool valid   = filled[p] == (int32_t) r;
                    const bool visible = p < bo_vis;

                    q_pool_bias[p] = valid && visible ? 0.0f : -INFINITY;
                }
            }
        }

        // exactly one partition per row, or a query reads another sequence's pools
        GGML_ASSERT(n_done == n_tps && "every query must belong to a sequence of the ubatch");

        if (kcache) {
            // the fixed row count means unused slots must name a safe destination: repeat a complete
            // pool (recompute is a no-op), or cell 0 when none exists (nothing reads its pooled third)
            for (int64_t p = n_new; p < n_new_max; ++p) {
                if (any_rep_src) {
                    std::copy(any_rep_src, any_rep_src + r, cur_new_cells + p*r);
                    cur_new_reps[p] = (int64_t) strm_of[s]*kv_size + any_rep_src[r - 1];
                } else {
                    cur_new_reps[p] = (int64_t) strm_of[s]*kv_size;
                }
            }
        }
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    // unconditional: the key/gate STORE runs on the dense path too, or cells below n_select
    // would have no indexer state when the first ubatch crosses it
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    if (pool_cells == nullptr) {
        return;
    }

    std::vector<uint32_t> strm_of;

    if (pool_reps) {
        strm_of.resize(mctx_idx->get_n_stream());

        for (uint32_t s = 0; s < strm_of.size(); ++s) {
            strm_of[s] = mctx_idx->get_strm(s);
        }
    }

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias,
            sel_mask, cand_mask,
            pool_reps, new_pool_cells, new_pool_reps,
            strm_of.empty() ? nullptr : strm_of.data(),
            pool_reps ? (int64_t) mctx_idx->get_kv()->get_size() : 0,
            rebuild,
            ubatch, kpool);

    // cleared here, not in build_inp_kpool: a graph built but not evaluated must not clear it
    if (rebuild) {
        mctx_attn->get_kv()->clear_kpool_dirty();
    }
}
