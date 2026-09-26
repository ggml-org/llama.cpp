# RQ-B Research Artifact: Correct TurboQuant Flash-Attention on SYCL (Arc A770)

_Read-only adversarial research. No source edited. Deliverable for /plan._

---

## 1. Framing pushback (Phase 0.5)

The brief frames the decision as **"make TILE correct" vs "route to VEC + complete tq2/tq4 instances,"** and attaches a prior draft whose verdict is **"DEFINITIVE NO-GO, the TILE path is the fix to pursue but ultimately keep the veto and use Vulkan."** Four parts of that framing are wrong or unverified at source, and I reject them:

1. **The TILE-vs-VEC axis is real but the prior draft picked the wrong pole.** TILE is not a viable turbo carrier on SYCL: every TILE launch hard-codes need_f16_K=need_f16_V=true [S10], which makes launch_fattn dequantize the WHOLE turbo KV tensor to a fresh f16 VRAM buffer [S13] -- defeating the entire point of KV compression -- AND the TILE kernel then re-reads that f16 buffer through the turbo block decoder [S11], a type confusion that yields garbage. VEC, by contrast, sets need_f16 = (type==F16) -> false for turbo [S7], decodes turbo inline [S14], and already has tq2/tq3/tq4 dequant helpers [S14]. The correct pole is VEC. The prior draft inverted this.

2. **"Complete tq2/tq4 instances" is largely a non-task.** The turbo VEC cases instantiate IMPLICITLY in fattn.cpp because the header intentionally does NOT extern-declare them [S8]. The dequant helpers exist for all three turbo widths [S14]. Missing instance .cpp files are a compile-parallelism nicety, not a correctness blocker. The brief and prior draft both overstate this.

3. **The double-WHT "gate the graph on !use_flash_attn" hypothesis is the wrong fix.** The graph rotation is correct and shared with the working non-FA path; the kernel rotation is the redundant, INCOMPLETE one (it omits the InnerQ scale_inv the graph applies [S5][S17][S22]). The fix is to delete the kernel rotation, not gate the graph. No build_attn_mha gate edit is needed.

4. **The NO-GO's load-bearing premise -- "Vulkan turbo FA is device-verified, just use it" -- is unverified at this commit.** The fork's own Vulkan host code says `turbo3 FA SPIR-V generation deferred; no dedicated pipeline yet` [S29]. The shader path is wired (graph-owns-WHT, dequantize4 centroid decode [S30][S31]) and supports_op advertises it [S28], but "device-verified" is not supported by source. A NO-GO that punts to an unverified backend is not a grounded NO-GO.

**Reframed question (targeted by the rest of this artifact):** *Given that the canonical turbo-FA contract (graph-owns-WHT, kernel-does-centroid-dequant-only) is already implemented and working on CUDA and wired on Vulkan, and that the SYCL VEC kernel already decodes turbo inline, what is the minimal correct change-set to route SYCL turbo KV through the VEC flash-attention path -- and is it worth doing vs the standing Vulkan recommendation?*

**Open question surfaced to the human (reframe check):** I narrowed B4 from an open TILE-vs-VEC contest to a committed VEC recommendation because the source forecloses TILE. If you specifically want TILE evaluated as the carrier (e.g. for prefill throughput, accepting f16 KV expansion), say so -- that is a different, larger project (see Implementation strategies / Escalation).

---

## 2. Version pin / frame block

| Field | Value |
|---|---|
| Repo | ~/projects/trb/llama-cpp-turboquant (private fork) |
| git HEAD | `fc3584d62c26db5aeaec3c8f4fe680916fcb1865` |
| git log -1 | `fc3584d62 sycl : fix q8_0 KV flash-attention crash on Arc A770` |
| Branch | `merge/sycl-turboquant` (clean: 0 staged/unstaged/untracked) |
| Compiler | Intel oneAPI DPC++/C++ (icpx/icx) 2026.0.0 (2026.0.0.20260331) |
| oneAPI roots present | 2025.0, 2025.3, 2026.0 (latest -> 2026.0) |
| Target GPU | Intel Arc A770 (Xe-HPG / DG2), KMD xe; per workstation profile |
| WARP size | GGML_SYCL_WARP_SIZE=16 (INTEL) [CMakeLists.txt] |

**Settled premises accepted from brief (not re-litigated):** Bug A (tile enum-truthiness gate) fixed [S9]; q8_0 decode crash fixed at HEAD [S33]; f16/q8_0 standard-KV FA works device-side; f32 is not an in-model FA path.

**Hard guardrails honored:** (a) net WHT rotations counted end-to-end with cited lines (B2). (b) test-backend-ops FLASH_ATTN_EXT NMSE is NOT cited as a turbo-FA verdict anywhere. (c) every load-bearing claim cites a file:line I personally opened in THIS fork (the prior round's three confabulations are explicitly corrected below). (d) every finding tagged [source-answerable] or [needs-A770].

**Prior-round confabulation corrections (verified at source this round):**
- Prior draft: "add routing logic to force turbo->TILE." FALSE -- turbo is ALREADY accepted by the type-switch [S2:231-233] and ALREADY routed to TILE [S2:259-262, 269]. No routing needs to be ADDED to reach TILE; routing needs to be CHANGED to reach VEC.
- Prior draft: "make TILE correct, keep VEC vetoed." Backwards -- TILE forces f16 expansion [S10][S13]; VEC is the compression-preserving path [S7].
- Subagent (Vulkan): "faDecodeK/V return float16_t(0) for turbo -> silently broken." REFUTED -- USE_DECODE_K is true for turbo and routes to dequantize4 [S31][S32]; the zero-fallback is the f16 branch, not taken for turbo.

---

## 3. B1-B5 answers

### B1 -- Change-site enumeration (every site that must change)

The SYCL FA selector already accepts turbo and routes it to TILE; only the explicit veto stops it. The minimal correct route is **VEC** (see B4). Sites:

**B1.1 Veto [source-answerable].** `ggml_sycl_flash_attn_ext_supported` returns false for any turbo K or V [S1: fattn.cpp:286-298], with a comment that itself documents the double-WHT mechanism ("the model graph already applies the TurboQuant WHT around attention ... so FA would double-apply it") [S1:289-294]. Edit: remove the turbo early-return so it falls through to `ggml_sycl_get_best_fattn_kernel(...) != NONE` [S1:297-299]. Effort Low / Risk Low. [needs-A770] to confirm it then produces correct output.

**B1.2 Routing [source-answerable + needs-A770].** `ggml_sycl_get_best_fattn_kernel` currently: the K-type switch already has `case TURBO2_0/3_0/4_0: break` (accepted) [S2:231-233]; for quantized K with Q->ne[1] <= 2 it does `switch(K->type){ TURBO -> return BEST_FATTN_KERNEL_TILE; default -> return VEC; }` [S2:255-264]; and the function tail unconditionally `return BEST_FATTN_KERNEL_TILE` [S2:269]. Edit: change the turbo arm to return VEC (or delete the turbo special-case so it hits `default: return VEC`), AND ensure prefill (Q->ne[1] > 2) turbo also returns VEC rather than falling to the TILE tail -- mirroring CUDA which forces quantized/turbo KV to VEC at Q->ne[1] <= 8 [S26:551-552]. Effort Low-Med / Risk Med (prefill perf). [needs-A770] for perf; [source-answerable] for correctness of the routing.

**B1.3 Missing tq2/tq4 VEC instances [source-answerable] -- NOT a blocker.** Only `fattn-vec-instance-tq3-tq3.cpp` exists [S35], but the header comment states turbo types are intentionally NOT extern-declared and are "instantiated implicitly in fattn.cpp's dispatch" [S8]; the dispatch references tq2-tq2/tq3-tq3/tq4-tq4 at D in {64,128,256,512} [S3]; the dequant helpers exist for all three [S14]; CMake GLOBs all instance files [S34]. So routing turbo->VEC compiles WITHOUT new instance files. OPTIONAL: add tq2-tq2.cpp + tq4-tq4.cpp (and mixed combos) to mirror CUDA's explicit instances [S25] for build parallelism. Effort Low / Risk Low.

**B1.4 Graph WHT gating [source-answerable] -- NO edit needed.** The graph forward-WHT on Q lives in the caller build_attn (gated only on k->type==turbo, NOT on use_flash_attn) [S22:2382-2391]; the inverse-WHT on output is applied in BOTH branches of build_attn_mha (FA branch gated v->type [S21:2107-2114], non-FA branch gated v->type [S21:2185-2192]). Because the graph already owns a correct, FA-independent rotation, and the fix removes the kernel rotation (B2), no `!use_flash_attn` gate is required. Editing the graph would be the WRONG fix (B2). Effort None.

**B1.5 launch_fattn need_f16 for turbo [source-answerable] -- NO edit needed on the VEC route.** The prior draft's "set need_f16=false or OOM at 262K" applies only to TILE, which hard-codes true [S10]; VEC sets need_f16 = (type==F16) -> false for turbo [S7:661-662], so VEC never triggers the full-tensor to_fp16 path [S13]. The 16 GiB-scratch OOM concern is a TILE problem we avoid by not using TILE. Effort None on VEC route.

**B1.6 Kernel-internal WHT removal [source-answerable + needs-A770] -- the real correctness edit (see B2).** Remove the turbo forward-WHT-on-Q block in the VEC kernel (both code paths: F16 [S5:249-291], non-F16 [S6:306-345]) so the VEC kernel consumes the already-graph-rotated Q and only inline-decodes K/V. Effort Med / Risk Med. [needs-A770] for final numeric confirmation; [source-answerable] for the math.

**B1.7 Dead TILE turbo path cleanup [source-answerable] -- AGENTS clean-cutover.** Once turbo routes to VEC, the turbo TILE cases [S12: fattn-tile.cpp:27-85] and the tile turbo Q-rotation [S9: fattn-tile.hpp:937-953] become unreachable. Remove them (no shims). Effort Low / Risk Low.

### B2 -- The double-WHT reconciliation (the crux), with net-rotation count

**WHT algebra.** The normalized transform R = (1/sqrt(D)) * diag(S2) * H * diag(S1) is orthonormal; for a Hadamard H with H*H = D*I, R is effectively self-inverse up to the sign-table swap the code uses for direction=1 [S17][S19][S23]. Orthonormality is what makes (R*Q).(R*K) = Q.K.

**Where each rotation lives (cited):**
- KV store: set_rows writes K/V as normalize -> S1 -> H -> S2 -> 1/sqrt(128) -> quantize, i.e. the cache holds R-rotated, quantized values [S20:78-84].
- Graph forward on Q: ggml_turbo_wht(q, dir=0, group=0, innerq_scale) in build_attn, gated k->type==turbo, BEFORE build_attn_mha, independent of use_flash_attn [S22:2382-2391].
- Graph inverse on output: ggml_turbo_wht(cur, dir=1, group, innerq_scale), FA branch [S21:2114] and non-FA branch [S21:2192].
- SYCL VEC kernel forward on Q (the offender): S1 -> turbo_wht<D> -> S2 -> 1/sqrt(D), with NO scale_inv [S5:249-291][S6:331-345].
- SYCL TILE kernel forward on Q (the offender): same shape [S9:937-953].
- CUDA VEC kernel: centroid LUT dequant only, NO WHT [S24]. Vulkan FA shader: dequant returns centroid*norm, NO WHT [S30].

**Net rotation count -- if veto is lifted with NO other change (turbo through current VEC/TILE):**
- Q: graph forward (1) [S22:2390] + kernel forward (1) [S5/S9] = W*W*Q = Q. Q ends UN-rotated.
- K (dequant): R*K (stored rotated) [S20].
- Scores = Q . (R*K)^T != Q.K^T -> mismatched basis -> garbage. Matches the veto comment [S1:289-294] and is the literal double-WHT landmine.
- Output: kernel computes P*(R*V) = R*O; graph inverse (1) [S21:2114] -> O. (Output side is single-rotation/correct, but P is already garbage.)
- Extra defect: kernel forward omits innerq scale_inv that the graph applies [S5 vs S22/S17] -> even single-rotation would be numerically off when InnerQ scaling is active.

**Net rotation count -- proposed fix (delete kernel rotation, keep graph, route VEC):**
- Q: graph forward (1, with scale_inv) [S22:2390] + kernel (0) = R*Q. Correct, single, includes InnerQ scale.
- Scores = (R*Q) . (R*K)^T = Q.K^T. Correct basis.
- Output: kernel P*(R*V) = R*O; graph inverse (1, with scale_inv) [S21:2114] -> O. Correct.
- NET: exactly 1 forward on Q (graph) + 1 inverse on output (graph) + 0 kernel rotations == the CUDA [S24] and Vulkan [S30] canonical contract.

**Which to remove:** remove the KERNEL rotation. **Minimal edit:** in flash_attn_ext_vec, make the `if constexpr (K_is_turbo)` Q-load branches [S5:249-291, S6:306-345] load Q exactly like the non-turbo f16/float branch (scaled copy into Q_reg), deleting the SIGNS1/turbo_wht/SIGNS2/(1/sqrt) steps. **Exact gate site the brief asked about (build_attn_mha):** the use_flash_attn split is at llama-graph.cpp:2080-2081 [S21]; the forward-WHT is OUTSIDE/upstream of it at 2382-2391 [S22]; the inverse-WHT is inside each branch at 2114 / 2192 [S21]. **No `!use_flash_attn` gate is added** -- doing so would leave the incomplete kernel rotation as the only FA rotation (missing scale_inv) and break the FA path; it would also not fix the basis mismatch. The non-FA turbo path is untouched because it never had a kernel rotation and the graph rotation it depends on is unchanged.

### B3 -- Canonical contract from a working backend, and the SYCL diff

I read TWO references at source (the brief named Vulkan as standing recommendation and CUDA-VEC as closest to SYCL-VEC); they agree.

**Vulkan contract [S28][S29][S30][S31][S32]:** WHT lives at the GRAPH (forward on Q, inverse on FA output); the FA shader does NOT rotate -- it dequantizes turbo blocks to centroid*norm inline via the spec-constant-switched dequantize4() uber-decoder. Verbatim contract statement: flash_attn_dequant.glsl:36-42 "Graph applies forward WHT to Q pre-attention and inverse WHT to FA output, so dequant just returns centroid * norm." Stored = R-rotated, quantized turbo blocks. supports_op allows TURBO2/3/4 for FLASH_ATTN_EXT [S28:16879-82]. Caveat: the turbo-specialized SPIR-V is "deferred; no dedicated pipeline yet" [S29]; the generic f16 pipeline + runtime FaTypeK decode is the intended carrier and is functional in source (USE_DECODE_K true for turbo -> dequantize4 [S31][S32:156]), but end-to-end device verification is [needs-A770/needs-Vulkan-trace].

**CUDA contract [S24][S25][S26]:** identical shape. fattn-vec.cuh treats turbo as "unquantized" for Q handling (no q8_1) [S24:88-90], scores via a shared-memory centroid LUT [S24:145-152, 280-345], decodes V as centroid*norm [S24:486-572], and contains NO WHT/hadamard/rotate anywhere in the kernel (the only `signs` are the per-block quantization sign byte at :328/:508, not the WHT S1/S2 tables) [S24]. Routing forces quantized/turbo KV -> VEC [S26:551-552]; support gate instantiates turbo VEC for D in {64,128,256} [S26:510-523]. HIP reuses the CUDA .cu sources (ggml-hip/ is only a CMakeLists [find]; GLOB of ../ggml-cuda/*.cu reported by subagent [S27]).

**Diff: SYCL turbo FA vs the canonical contract.**
| Aspect | Canonical (CUDA/Vulkan) | SYCL today | Action |
|---|---|---|---|
| WHT owner | graph only | graph + kernel (double) [S5][S9] | delete kernel rotation [B1.6] |
| Q to kernel | pre-rotated R*Q | re-rotated to ~Q [S5] | stop re-rotating |
| InnerQ scale on Q | applied by graph [S22] | omitted by kernel [S5] | fixed by deleting kernel rotation |
| K/V decode | inline centroid*norm | VEC inline OK [S14]; TILE forces f16 [S10][S13] | route VEC, drop TILE [B1.2/B1.7] |
| Carrier kernel | VEC (quantized) | TILE [S2] | route to VEC [B1.2] |
| KV compression kept during attn | yes | VEC yes / TILE no | VEC keeps it |

### B4 -- TILE vs VEC: committed to VEC (route turbo->VEC), TILE rejected

**Decision: route turbo -> VEC; do NOT pursue "make TILE correct."** Evidence:

1. **TILE structurally expands KV to f16.** Every TILE launch passes need_f16_K=need_f16_V=true [S10:1187-1245]; launch_fattn then allocates K_f16/V_f16 of ggml_nelements(K/V) and runs ggml_get_to_fp16_sycl(turbo) -- which EXISTS [S15:719-724] -- over the whole tensor [S13:1011-1069]. That is a full-precision KV scratch allocation per FA call: it negates KV compression and scales with context (the prior draft's ~16 GiB-at-262K concern is mechanistically real FOR TILE [source-answerable mechanism; needs-A770 for the exact OOM threshold]).
2. **TILE turbo is additionally type-confused.** After need_f16 converts turbo->f16, the TILE kernel (instantiated type_K=TURBO* [S12]) still dispatches the turbo block decoder over that f16 buffer [S11:399-411] -> reads f16 bytes as turbo blocks -> garbage. "Making TILE correct" means resolving this need_f16-vs-inline contradiction AND removing the kernel WHT -- strictly more work than the VEC route, for a path that defeats compression.
3. **VEC is already the compression-preserving, contract-correct carrier.** need_f16=false for turbo [S7], inline tq2/tq3/tq4 decode helpers exist [S14], cases instantiate implicitly [S8]. Only the kernel WHT [S5][S6] and the routing/veto [S1][S2] stand between today and a correct VEC turbo path. This mirrors CUDA/HIP [S26] -> AGENTS-compatible reuse.

**Register-footprint angle [source-answerable + needs-A770].** The HEAD commit is the key evidence [S33]: the q8_0 crash was the fork's `nthreads = max(default=128, D)` giving the unused D=256 q8_0 instance nthreads=256, which "spills the A770 register file in the F16=OFF build and fails the device-module compile." The fix reverts nthreads to a fixed 128 [S33: fattn-vec.hpp:103]. Consequences for turbo VEC: (a) at fixed nthreads=128 the D=256 spill that triggered the q8_0 crash no longer occurs; (b) deleting the kernel WHT [B1.6] removes the per-thread butterfly temporaries + SIGNS work, REDUCING turbo VEC footprint below today's; (c) turbo decode is a small centroid LUT (4/8/16 entries) [S24:149-152] kept in shared memory, not a large register array. Net: the prior draft's "tq2/tq4 VEC will spill" is not supported by source -- the evidence points to turbo VEC at D=128 (the common turbo head_dim) fitting comparably to the now-working q8_0 D=128. D=256 turbo VEC at nthreads=128 is plausible but [needs-A770] to confirm. Note the SYCL VEC does NOT replicate CUDA's turbo-specific nthreads_KQ=1 tuning [S24:94] (SYCL uses the generic quantized config [S4]); if A770 shows pressure at D=256, importing that tuning is the lever.

### B5 -- WHT-kernel consistency across the three (now four) definitions

All definitions resolve to the same forward shape `(scale_inv?) -> S1 -> H -> (1/sqrt(group)) -> S2` and inverse `S2 -> H -> (1/sqrt) -> S1 -> (scale_inv?)` [S23 confirms the API semantics: dir 0 = signs1->WHT->signs2, dir 1 = signs2->WHT->signs1].

| Element | Graph SYCL op [S17 turbo-wht.cpp] | FA kernel [S5/S6/S9] | CPU ref [S19 ops.cpp] | Store [S20 set_rows] |
|---|---|---|---|---|
| S1 table | TURBO_WHT_SIGNS1 [S16:58] | TURBO_WHT_SIGNS1 [S5:254] | turbo_wht_s1 [S19:10830] | TURBO_WHT_SIGNS1 [S20:79] |
| S2 table | TURBO_WHT_SIGNS2 [S16:69] | TURBO_WHT_SIGNS2 [S5:287] | turbo_wht_s2 [S19:10831] | TURBO_WHT_SIGNS2 [S20:82] |
| Butterfly | turbo_wht<group> [S18] | turbo_wht<D> [S5:285] | inline loop [S19:10882] | turbo_wht<128> [S20:81] |
| Norm (D=128) | 0.08838834764831845 = 1/sqrt(128) [S17:50] | 1/sqrtf(D) [S5:288] | 1/sqrtf(group) [S19:10857] | 1/sqrt(128) [S20:83] |
| Norm (D=64) | 0.125 = 1/sqrt(64) [S17:50] | 1/sqrtf(D) [S6] | 1/sqrtf(group) [S19] | n/a (128 store) |
| scale_inv | yes (fwd pre / inv post) [S17:35-67] | NO [S5][S6][S9] | yes (fwd pre / inv post) [S19:10873/10898] | per-group norm [S20:76] |
| direction select | s1/s2 by dir [S17:40-45,55-63] | forward only [S5] | s_first/s_second by dir [S19:10862-63] | forward only [S20] |

**SIGNS tables are bit-identical** between SYCL [S16] and CPU [S19]: I compared the first 40 of 128 entries element-for-element for both S1 and S2 and they match exactly (both generated from "seed=42" [S16:57]; ops.cpp comment: "must match Metal shader turbo_wht_signs1/2" [S19:10829]). The graph SYCL op and the FA kernel use the SAME C++ symbol from the same header (turbo-quants.hpp) [S16], so they are one definition -- trivially consistent. The D=64 variants (_64) exist in SYCL [S16:80-95] but the CPU reference uses "first 64 of the 128 array" [S19:10861 comment]; SYCL's _64 arrays are the first 64 entries of the 128 arrays [S16] -- consistent.

**Normalization is single, not doubled:** each definition applies 1/sqrt(group) exactly once [S17:50][S5:288][S19:10894][S20:83].

**Latent-bug flags:**
- **(Flag 1, real) Kernel rotation omits InnerQ scale_inv.** Graph fwd/inv pass innerq_scale [S22:2390][S21:2114]; the FA kernel rotation has no scale_inv term [S5][S6][S9]. For models with active InnerQ scaling this is a numeric divergence -- but it is MOOT under the fix because we delete the kernel rotation. Flagging because it confirms the kernel rotation was never a complete substitute for the graph rotation.
- **(Flag 2, watch) D=64 turbo coverage.** CPU group is parameterized; the FA kernel hard-branches D==64 vs else [S5:253-254]. Consistent today, but any future head_dim that is a multiple of 64 but not 128 routes through _64 tables in the kernel while the graph chooses group=64 only if ne[0]%128 != 0 [S22:not shown / turbo_group logic S21:2110]. Low risk; note for the planner.
- No SIGNS-pattern or normalization divergence found among the four definitions.

---

## 4. Ranked file:line change-set

| # | File:line | Block | Action | Effort/Risk | Device-dep | AGENTS |
|---|---|---|---|---|---|---|
| 1 | ggml-sycl/fattn-vec.hpp:249-291, 306-345 | K_is_turbo Q-rotation (F16 + non-F16) | Delete S1/turbo_wht/S2/(1/sqrt); load Q as plain scaled copy like the f16 branch | Med / Med | [source-answerable] math; [needs-A770] numerics | Yes (fix, reuse) |
| 2 | ggml-sycl/fattn.cpp:255-269 | get_best_fattn_kernel turbo arm + TILE tail | Route turbo->VEC for decode AND prefill (mirror CUDA quantized->VEC) | Low-Med / Med | [needs-A770] perf | Yes |
| 3 | ggml-sycl/fattn.cpp:286-298 | ggml_sycl_flash_attn_ext_supported | Remove turbo false-return; fall through to selector | Low / Low | [needs-A770] enables path | Yes |
| 4 | (none) src/llama-graph.cpp:2382-2391, 2107-2114, 2185-2192 | graph WHT | NO EDIT -- graph already owns correct, FA-independent rotation | None / None | [source-answerable] | Yes |
| 5 | (none) ggml-sycl/fattn-common.hpp:1011-1069 | launch_fattn need_f16 | NO EDIT on VEC route -- VEC need_f16=false for turbo [S7] | None / None | [source-answerable] | Yes |
| 6 | ggml-sycl/fattn-tile.cpp:27-85; fattn-tile.hpp:937-953 | dead turbo TILE cases + tile WHT | Remove (clean cutover; unreachable once routed to VEC) | Low / Low | [source-answerable] | Yes |
| 7 | ggml-sycl/template-instances/ (+CMake GLOB) | tq2-tq2.cpp, tq4-tq4.cpp (+ mixed) | OPTIONAL add to mirror CUDA explicit instances for compile parallelism (implicit instantiation already works) | Low / Low | [source-answerable] | Yes |
| 8 | tests/test-sycl-turbo-correctness.cpp | probes | Add head_dim=128 turbo VEC vs CPU-cosine probe + e2e generation (NOT NMSE 5e-4) | Low / Low | [needs-A770] to run | Yes |

---

## 5. Committed decisions

**WHT-ownership: GRAPH-OWNS (delete kernel rotation).** The graph forward-on-Q [S22] + inverse-on-output [S21] is the contract both references implement [S24][S30] and is shared by the working SYCL non-FA path. The kernel rotation [S5][S6][S9] is redundant (double-WHT [B2]) and incomplete (no InnerQ scale_inv [B5 Flag 1]). What would change my mind: if a turbo model existed with NO graph WHT wiring (it does not -- the gates are unconditional on k/v type [S22][S21]).

**TILE vs VEC: VEC.** TILE expands turbo to f16 [S10][S13] and is type-confused [S11]; VEC preserves compression and already decodes turbo inline [S7][S14], matching CUDA/HIP [S26]. What would change my mind: A770 measurements showing VEC prefill is catastrophically slow AND a willingness to spend f16 KV scratch (then escalate to a TILE-as-f16 prefill path, B4 escalation).

**Overall verdict: CONDITIONAL GO (implement via VEC, then device-gate) -- NOT a permanent NO-GO.** Reasoning: the canonical contract is verified and triply-corroborated [S24][S30][S21]; the SYCL VEC path is ~3 edits from correct [changes 1-3]; the prior draft's two pillars collapse under source (TILE is wrong; Vulkan "device-verified" is unconfirmed [S29]). The residual risk is entirely A770 numeric/perf behavior, which is exactly what the veto can keep guarding until a device run passes. So: implement on a branch, verify on A770 with a head_dim-128 cosine probe + e2e generation, keep the veto as the documented fallback if numerics fail. This is a tractable, AGENTS-compatible (reuse VEC infra, mirror CUDA, no new subsystem) change -- it is PAUSE-worthy only for change #1 (touches kernel math) and should be gated on device verification before the veto is removed in a shipping build.

---

## 6. Adversarial self-attack (Phase 4)

**Where I could still be wrong, and the falsifier for each:**

1. **"Deleting the kernel rotation is sufficient."** Risk: the VEC turbo Q-load path may have a second dependency on the rotation (e.g. Q_reg layout assumptions, the KQ shared-memory reuse). Falsifier: after the edit, a head_dim-128 turbo VEC run still mismatches CPU cosine despite single-net-rotation. Mitigation: the f16 non-turbo branch is the exact load template to copy [S5 else-branch]. [needs-A770].
2. **"VEC handles prefill correctly."** Risk: cols_per_block is capped at 2 [S7 ext_vec_case], and the launch grid for large Q->ne[1] may have an untested path for turbo. Falsifier: prefill (Q->ne[1] large) turbo produces wrong logits while decode is correct. Mitigation: CUDA runs the same VEC-for-quantized design [S26]; still [needs-A770].
3. **"Vulkan is a real fallback."** Risk: I did NOT confirm a turbo FA pipeline is actually registered/dispatched at runtime; the "deferred; no dedicated pipeline yet" comment [S29] may mean turbo FA never dispatches (falls to non-FA or asserts). Falsifier: a Vulkan run with -fa on + turbo KV either errors "no pipeline" or silently runs non-FA. This is why I downgraded the NO-GO premise rather than relying on it. [needs-A770/needs-Vulkan-trace].
4. **"Register footprint fits."** Risk: turbo decode adds LUT + index math the q8_0 path lacks; at D=256 with nthreads=128 it could still spill at compile in F16=OFF. Falsifier: device-module compile failure for a tq*-tq* D=256 instance (same signature as the q8_0 crash [S33]). Mitigation: import CUDA's turbo nthreads tuning [S24:94]; or restrict turbo VEC to D in {64,128}. [needs-A770].
5. **"SIGNS tables match end-to-end."** Risk: I verified 40/128 entries, not all 128, and did not diff the _64 arrays exhaustively. Falsifier: a byte-diff of entries 40-127 of S1/S2 between turbo-quants.hpp [S16] and ops.cpp [S19] disagrees. Mitigation: trivial to complete (a full array diff); the seed=42 provenance and shared-symbol usage make divergence unlikely. [source-answerable -- a remaining lead].
6. **"The math (W self-inverse) holds for the code's S1/S2 asymmetry."** Risk: forward uses S1-then-S2 while inverse uses S2-then-S1 [S17][S19]; if S1 != S2 (they differ [S16]), round-trip correctness depends on the specific construction. Falsifier: a CPU round-trip turbo_wht(dir0) then turbo_wht(dir1) on random input not returning identity. Mitigation: the working non-FA path and CPU reference exercise exactly this round-trip in production [S20 store + S21 inverse]; [source-answerable] via a 1-hour CPU test.

**Convergence check:** I did NOT land on the brief's framing -- I inverted its TILE/VEC pole and downgraded its NO-GO. The inversion was driven by source (need_f16 hard-coding [S10], implicit instantiation [S8], the Vulkan "deferred" comment [S29]), not by preference. I ran the inversion twice (TILE-as-f16-correct is acknowledged as a real but compression-defeating escalation, not dismissed).

---

## 7. Meta-observation

Confidence: HIGH on the static contract and change-set (B1, B2, B3, B5 -- all source-cited and cross-checked across CUDA + Vulkan + SYCL non-FA + CPU). MEDIUM-to-conditional on the GO verdict, because every remaining unknown is device-resident (A770 numerics, VEC prefill perf, D=256 register fit, whether Vulkan turbo FA actually dispatches). This research is structurally vulnerable to exactly the failure that bit the prior round and one of my own subagents: a fluent, plausible claim about code that is wrong (the Vulkan "returns zero" confabulation [refuted by S31][S32]). The guard I applied -- and that /plan must keep applying -- is: treat every subagent finding as a lead, re-open the file, and count mechanism (here: net WHT rotations [B2]) rather than trust a slogan. The LLM bias most dangerous to THIS design is the pull toward a clean NO-GO narrative ("SYCL is hopeless, use Vulkan") because it is tidy and defensible-sounding; the source does not support it, and the honest answer is the messier "3 small edits + a device gate." The design's own safeguard is the veto itself: it is a perfect kill-switch, so the implementation can be landed behind it and only un-vetoed once an A770 cosine + e2e probe passes -- the code carries its own falsification gate.

**What I would verify next on an A770 (in order):**
1. Apply changes 1-3 on a branch; run a head_dim-128 turbo3 VEC FA vs CPU cosine probe (NOT test-backend-ops NMSE 5e-4, which over-reports [guardrail]).
2. e2e generation (turbo3 KV, -fa on) vs turbo3 KV -fa off (non-FA) -- compare perplexity/text.
3. Compile-check tq2/tq4 D=256 VEC instances in F16=OFF (the q8_0 crash signature [S33]).
4. Decode and prefill latency: turbo VEC FA vs turbo non-FA vs f16 FA.
5. Independently, settle whether Vulkan turbo FA actually dispatches (pipeline registration trace) to confirm/deny the fallback.

---

## Sources (every entry personally opened in THIS fork unless tagged)

- [S1] ggml/src/ggml-sycl/fattn.cpp:286-299 -- ggml_sycl_flash_attn_ext_supported (turbo veto + mechanism comment)
- [S2] ggml/src/ggml-sycl/fattn.cpp:148-270 -- ggml_sycl_get_best_fattn_kernel (type-switch 231-233; turbo->TILE 259-262; tail TILE 269)
- [S3] ggml/src/ggml-sycl/fattn.cpp:21-148 -- FATTN_VEC_CASES_ALL_D incl turbo (default branch tq2/tq3/tq4)
- [S4] ggml/src/ggml-sycl/fattn-vec.hpp:108-115 -- nthreads(=128), K_is_turbo, Q_q8_1
- [S5] ggml/src/ggml-sycl/fattn-vec.hpp:249-291 -- F16-path turbo Q rotation (S1 254 / WHT 285 / S2 287 / 1sqrt 288)
- [S6] ggml/src/ggml-sycl/fattn-vec.hpp:306-345 -- non-F16-path turbo Q rotation
- [S7] ggml/src/ggml-sycl/fattn-vec.hpp:655-668 -- ext_vec_case_impl need_f16_K/V (661-662), launch_fattn call
- [S8] ggml/src/ggml-sycl/fattn-vec.hpp:700-747 -- EXTERN_DECL_FATTN_VEC_CASES + comment: turbo NOT extern-declared (implicit instantiation)
- [S9] ggml/src/ggml-sycl/fattn-tile.hpp:932-953 -- tile turbo Q rotation + BUG-FIX (enum-truthiness) comment
- [S10] ggml/src/ggml-sycl/fattn-tile.hpp:1187-1245 -- launch_fattn calls all pass need_f16_K=need_f16_V=true
- [S11] ggml/src/ggml-sycl/fattn-tile.hpp:399-411 -- tile turbo load dispatch by type_K (turbo block decoders)
- [S12] ggml/src/ggml-sycl/fattn-tile.cpp:11-95 -- host tile dispatch; turbo tile_cases at D 64/128/256/512
- [S13] ggml/src/ggml-sycl/fattn-common.hpp:963-1069 -- launch_fattn; need_f16 full-tensor to_fp16 (1011-1035 K, 1038-1069 V)
- [S14] ggml/src/ggml-sycl/fattn-common.hpp:610-678 -- dequantize_V_turbo_generic; get_vec_dot_KQ turbo (647-652); get_dequantize_V turbo (673-678)
- [S15] ggml/src/ggml-sycl/convert.cpp:719-724 -- ggml_get_to_fp16_sycl turbo registration
- [S16] ggml/src/ggml-sycl/turbo-quants.hpp:58-95 -- TURBO_WHT_SIGNS1/2[128] + _64[64] (seed=42)
- [S17] ggml/src/ggml-sycl/turbo-wht.cpp:11-69 -- k_turbo_wht_f32_sycl (fwd/inv, 1/sqrt 50, S2 on fwd 57)
- [S18] ggml/src/ggml-sycl/turbo-wht.hpp:28-60 -- turbo_wht<> butterfly (pure Hadamard, subgroup-shuffle)
- [S19] ggml/src/ggml-cpu/ops.cpp:10827-10916 -- CPU ref WHT; turbo_wht_s1/s2 (10830-31), body (10862-10905)
- [S20] ggml/src/ggml-sycl/set_rows.cpp:54-103 -- KV store rotation (norm, S1 79, WHT 81, S2 82, 1/sqrt128 83, quantize)
- [S21] src/llama-graph.cpp:2079-2203 -- build_attn_mha (use_flash_attn 2080; FA inverse-WHT 2107-2114; non-FA inverse-WHT 2185-2192)
- [S22] src/llama-graph.cpp:2359-2412 -- build_attn caller (forward-WHT-on-Q gate 2382, call 2390; output extract 2399+)
- [S23] ggml/include/ggml.h:2567-2585 -- ggml_turbo_wht API + direction/normalize comment
- [S24] ggml/src/ggml-cuda/fattn-vec.cuh:88-110,145-152,280-345,486-572 -- CUDA turbo VEC centroid dequant, NO WHT (signs at 328/508 are quant sign bytes)
- [S25] ggml/src/ggml-cuda/fattn-vec.cuh:844-937 -- CUDA explicit turbo instances (all combos, D 64/128/256)
- [S26] ggml/src/ggml-cuda/fattn.cu:339-382,484-552 -- quantized/turbo->VEC (551-552); support gate (510-523); can_use_vector_kernel (538)
- [S27] ggml/src/ggml-hip/CMakeLists.txt -- HIP GLOBs ../ggml-cuda/*.cu (subagent-reported; ggml-hip/ dir personally confirmed to contain only CMakeLists.txt via find)
- [S28] ggml/src/ggml-vulkan/ggml-vulkan.cpp:16845-16890 -- FA supports_op allows TURBO2/3/4 (16879-82) + fused-dequant comment
- [S29] ggml/src/ggml-vulkan/ggml-vulkan.cpp:3949-4002 -- "turbo3 FA SPIR-V generation deferred; no dedicated pipeline yet" (x2), (void)is_turbo3
- [S30] ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl:1-42 -- uber dequantize4 + canonical contract comment (graph WHT; dequant returns centroid*norm)
- [S31] ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp:254-272 -- USE_DECODE_K->dequantize4 for turbo (zero-fallback is the f16 else-branch)
- [S32] ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl:95-108,156-157 -- fa_block_elems turbo; USE_DECODE_K/V defs
- [S33] git show fc3584d62 (HEAD) -- q8_0 crash fix: nthreads max(default,D)->default(128); main-loop barrier revert; turbo-FA veto comment added
- [S34] ggml/src/ggml-sycl/CMakeLists.txt:28-31 -- GLOB template-instances/fattn-vec*.cpp (auto-compiled)
- [S35] ggml/src/ggml-sycl/template-instances/fattn-vec-instance-tq3-tq3.cpp -- only turbo VEC instance present (tq3 only)

## Verification leads (low-confidence / device-gated)
- Full 128-entry byte diff of TURBO_WHT_SIGNS1/2 [S16] vs turbo_wht_s1/s2 [S19] (I verified 40/128). [source-answerable, ~5 min]
- CPU round-trip identity test: turbo_wht dir=0 then dir=1 == input. [source-answerable, <1 hr]
- Vulkan: confirm whether pipeline_flash_attn_f32_f16 actually contains turbo K-type keys at runtime (resolve the "deferred" ambiguity [S29]). [needs-Vulkan-trace]
- A770: head_dim-128 turbo3 VEC FA vs CPU cosine; e2e -fa on/off perplexity; tq2/tq4 D=256 F16=OFF compile. [needs-A770]

## Parking lot (out of scope this round)
- MLA turbo FA (v_mla path [S21:2118-2135]) -- the V=512/K=576 group-size-from-K logic [S21:2110] needs its own analysis.
- Mixed turbo/q8_0 and turbo/f16 KV combos (CUDA instantiates them [S25]; SYCL dispatch references them [S3]) -- correctness of asymmetric K!=V turbo not separately verified here.
- D=64 turbo head sizes (_64 sign tables [S16]) -- coverage noted (B5 Flag 2), not exercised.
- Vulkan turbo FA completion (generate the deferred SPIR-V variant [S29]) -- separate Vulkan workstream.
- Whether turbo non-FA already meets long-context perf needs (would reduce urgency of turbo FA entirely) -- product question for the human.
