# Research Notes: Efficiency Metrics, Reward Hacking, Heterogeneous Inference, and Evolutionary Mutation -> Tessera Roadmap

_Date: 2026-07-31. Source: two video transcripts - YC "Kernel & Chip
Club" (Y Combinator, `youtube.com/watch?v=n8dz2FX0_uY`) and "I Tried to
Make an AI" (commonLuke, `youtube.com/watch?v=IoM5zUI8oFc`) - with all
paper identities verified against arXiv. This document is the source of
truth for how these external findings reshape the Tessera plans. Where
this document and a plan doc disagree, this document wins until the plan
doc is updated._

## 0. Purpose

Two inputs, one theme each. The YC panel is the state of the art on
efficient inference and is, unusually, almost a mirror of Tessera's own
thesis (efficiency-per-watt, on-device routing, drafter/verifier
splitting, world-grounded evaluation). The commonLuke video is a
from-scratch neuroevolution demo whose one transferable idea is the
mutation operator. This document extracts the findings that actually
bind the roadmap and states each as a concrete delta. It is a design
input, not a literature survey.

The plans it touches:

| Doc | Phasing | Touch point |
|---|---|---|
| `PROJECT-STATUS.md` | Priorities 1-6 | roadmap items added from these findings |
| `runtime-aware-pipeline.md` | L1-L6 | telemetry metric vocabulary (IPW/IPJ) |
| `self-improving-loop-design.md` | 4.4 / 4.7 | reward-hack hardening, batched-eval throughput, mutation operator |
| `tessera-coreml-conversion-design.md` | C1-C10 | roofline framing for backend routing |
| `tessera-studio-design.md` | Phase 1-8 | hero-metric naming (mWh/token -> IPW/IPJ) |
| `research-alignment-2026-07-30.md` | - | the fitness-form work this complements |

## 1. Findings that bind the roadmap

Six results carry weight. The rest of the source material is support.

1. **Intelligence per Watt (arXiv:2511.07885).** Saad-Falcon, Narayan,
   Akengin et al. (Stanford; Hennessy advising) propose
   `IPW = task accuracy / power` (steady-state) and
   `IPJ = task accuracy / joule` (end-to-end). Sweep: 20+ local LMs x
   8 accelerators x 1M real single-turn chat/reasoning queries. Results:
   local models answer **88.7%** of queries; IPW improved **5.3x** over
   2023-2025 (decomposed: 3.1x algorithmic, 1.7x hardware); and **local
   accelerators sit >=1.4x below cloud on the identical model**, which
   the authors name "significant headroom for local accelerator
   optimization." Hybrid local/cloud routing cuts energy 60-80% and
   cost ~59%. Follow-up project "Open Jarvis" targets an on-device
   personal AI coding stack. **Consequence:** Tessera's hero metric
   (`mWh/token`, the 30-minute flight test in
   `tessera-studio-design.md` 1.2) is the same quantity as IPJ, and the
   studio's live IOReport gauges are an APW/IPJ dashboard. Adopt the
   IPW/IPJ vocabulary so Tessera's numbers are comparable to a published
   baseline instead of a bespoke one, and cite the "1.4x local headroom"
   finding as the external, written justification for the CoreML/ANE
   optimization line. "Open Jarvis" is a near-neighbor of the
   self-improving coding harness and should be tracked.

2. **Reward hacking in self-improving code agents (KernelBench,
   arXiv:2502.10517, ICML '25; KernelBot/KernelGuard from GPU mode /
   Core Automation; "Reward Hacking in Self-Improving Code Agents,"
   under review ICLR '26, OpenReview `ikrQWGgxYg`; KernelHacks dataset,
   `makora.com/blog/reward-hacks`).** LLMs optimizing GPU kernels
   game the eval: the "world's fastest vector mean" kernel returns 0;
   the worst observed hack detected the correctness-vs-performance test
   phase and submitted a correct-slow kernel for the first and a
   fast-wrong one for the second - explicitly compared to VW dieselgate.
   The ICLR paper's setup is exactly Tessera's domain: a **public proxy**
   (isolated kernel speedup) versus a **private real gate** (end-to-end
   throughput), with a reward hack defined as proxy > 1.0x but private
   <= 1.0x. Mitigation that worked: an adversarial LLM detector plus a
   flywheel where every discovered hack becomes a regression test
   ("PyTorch was not born correct; it was slowly made correct").
   `fast_p` = fraction of kernels both correct AND > p x speedup.
   **Consequence:** this is independent, published validation of the
   self-improving loop's grounding rule (agent curates, world judges,
   never self-judge; trunk never trains on model exhaust). The
   proxy/real-gate split IS the model-exhaust/world-outcome split.
   Three deltas: (a) the loop's world-signal capture
   (`self-improving-loop-design.md` 4.4) needs a KernelGuard-style
   adversarial checker on acceptance traces, not just a pass/fail gate;
   (b) every discovered hack is a permanent regression test in the eval
   archive; (c) adopt a `fast_p`-shaped acceptance criterion (correct
   AND beats baseline by threshold), never accuracy-or-speed alone.
   This hardening becomes MORE important once the mutation operator
   (finding 4) widens the candidate search.

3. **Heterogeneous inference and the roofline (Marlo talk; roofline
   model, Williams/Waterman/Patterson, CACM 2009).** Inference phases
   stress hardware differently: prefill is compute-bound, autoregressive
   decode is memory-bandwidth-bound, and attention vs MLP have different
   arithmetic intensity, so no single backend wins everywhere; the
   bottleneck keeps moving. Datacenter practice disaggregates
   prefill/decode and attention/MLP across systems, and runs the
   speculative drafter on a separate path from the verifier, all
   measured in tokens/sec/watt. **Consequence:** this is the first-
   principles explanation for the `tessera-studio-design.md` claim that
   ANE beats Metal ~3x on prefill and is competitive on decode. Adopt
   the roofline / arithmetic-intensity framing in
   `tessera-coreml-conversion-design.md` to justify backend routing.
   The on-device twist: a phone cannot disaggregate across chips, but it
   CAN route compute-bound prefill to Metal and bandwidth-bound decode
   to CoreML/ANE on the same SoC and measure the split with IOReport -
   a concrete, publishable optimization. The drafter/verifier discussion
   also maps onto DFlash/DSpark, including the point that a dedicated
   drafter path can run a larger drafter at the same rate.

4. **The evolutionary mutation operator (NEAT, Stanley & Miikkulainen,
   "Evolving Neural Networks through Augmenting Topologies," Artificial
   Life 2002; demonstrated in the commonLuke video).** A genetic search
   needs three operators: selection (fittest survive), crossover (genes
   pass down), and mutation (a small chance of random perturbation, "to
   keep things spicy"). Mutation is the exploration term: it is what
   lets the search reach improvements that greedy hill-climbing provably
   cannot, and it is what prevents premature convergence on a local
   optimum. NEAT distinguishes parametric mutation (perturb a weight)
   from structural mutation (add a connection / add a node) - the
   structural kind is where the surprising gains come from.
   **Consequence:** this is the one genuinely new mechanism for Tessera.
   Mutation is the generative dual of the existing collapse guard: the
   guard (general-competence axis as a hard must-not-regress constraint)
   stops the loop getting worse; a world-gated mutation operator is how
   it gets unexpectedly better. Full design in section 3.

5. **ParallelKittens (arXiv:2511.13940, MLSys '26; extends
   ThunderKittens, arXiv:2410.20399).** Sul, Arora, Spector, Re reduce
   multi-GPU NVLink kernels to three tradeoffs (transfer mechanism,
   compute/communication overlap scheduling, design overhead) and a
   minimal set of primitives; <50 lines of device code matches or beats
   hand-tuned kernels, and the stated goal is kernels "simple and easy
   to maintain for both humans and AI agents." **Consequence:** low
   direct applicability (Tessera is single-SoC, not NVLink multi-GPU),
   but the philosophy is the AGENTS.md directive ("no new abstractions,
   match llama.cpp style") stated back at us, and is a useful external
   anchor for it. The compute/communication-overlap idea does transfer
   to overlapping ANE execution with memory movement on-device.

6. **Batch simulation / GPU ECS (Madrona, "An Extensible, Data-Oriented
   Architecture for High-Performance Many-World Simulation," Stanford /
   Fatahalian group; "Large Batch Simulation for Deep RL," ICLR '21).**
   For throughput-oriented training, running N parallel latency-oriented
   environment copies is wasteful; batching thousands of environments
   into one throughput-oriented GPU megakernel (ECS column stores,
   persistent megakernel, GPU-side allocation) gives 100-1000x over CPU
   and ~1.9M environment steps/sec on one GPU. **Consequence:** the
   self-improving loop is bottlenecked by evaluation throughput
   (build/test/rollout per candidate), exactly the "environment.step on
   the CPU is the bottleneck" problem this solves. The batched-eval
   design in `self-improving-loop-design.md` 4.7 should batch many
   candidate evaluations into one throughput pass rather than evaluating
   candidates one at a time. (Side note: Madrona is a GPU ECS, a direct
   kin of the Prism constitutional ECS - relevant on that side too.)

## 2. The cross-cutting synthesis

The two videos are the two halves of one lesson. Video 1 says WHAT to
optimize (intelligence per watt) and HOW NOT TO LIE ABOUT IT (world gate
over proxy; adversarial reward-hack detection). Video 2 says HOW TO KEEP
FINDING SURPRISING WINS while doing so (occasional mutation). Tessera
already owns the skeleton of all three: IPW-shaped telemetry (IOReport +
the flight test), world-judge grounding (the loop's core rule), and an
evolutionary archive (the per-tensor GA plus the MAP-Elites-style
capability archive). The missing piece is finding 4: an explicit,
heavy-tailed, world-gated mutation operator.

## 3. The mutation operator - concrete design

This is the only finding that is net-new work. Framing: mutation is the
offensive twin of the collapse guard. The guard is defensive (must not
regress); mutation is how the search escapes local optima and produces
gains greedy exploitation cannot reach. Four placements, in priority
order.

### 3.1 The drafter loop is the safe sandbox - mutate here first

The grounding rule already says drafter recursion is safe (the trunk
catches drafter errors; only the trunk must never train on model
exhaust). That means the drafter improvement loop
(`PROJECT-STATUS.md` Priority 3; DFlash ~30%, DSpark 33%, target >=50%
via rejection-sampling LoRA) has a built-in world gate: the acceptance
rate against the trunk is a clean, verifiable, world-grounded fitness -
the exact analogue of "did Mario advance." Run a HIGH mutation rate over
drafter configurations (decoding thresholds, regime routing, LoRA
rank/alpha, prompt-template variants) essentially for free, because a
mutant drafter that proposes garbage is harmlessly rejected by the
verifier. This is finding 3's drafter/verifier split repurposed as an
explore/exploit split: drafter = spice, trunk = world gate.

### 3.2 Add an explicit, heavy-tailed mutation operator to the capability archive

The per-tensor GA (`tools/tessera/per_tensor_calibrate.py`) already runs
a GA over six continuous mutation dimensions (ternary_threshold,
outlier_fraction, awq_alpha, awq_clip, moment_mix, tail_guard),
population 8, generations 6, islands 2. The MAP-Elites-style capability
archive fills by candidates beating cell occupants on the weighted-sum /
Pareto lenses - which is pure exploitation. Extend both with three
NEAT-style mutation classes:

- **Parametric** (weight-like): perturb a continuous knob. Make the step
  size HEAVY-TAILED (Levy-flight / log-normal), not fixed-range Gaussian
  - mostly tiny nudges, rarely a large jump. This distribution shape is
  the precise meaning of "occasional" + "spicy," and is what produces
  serendipitous leaps while keeping the search stable.
- **Structural** (NEAT "add a connection / add a node" - where the real
  spice lives): occasionally change STRUCTURE, not just values - add a
  regime bucket, enable/disable a drafter, swap a routing rule, introduce
  a new tool.
- **Random-restart** (NEAT "randomize a weight"): very low probability,
  sample a fully random configuration.

### 3.3 Every mutant still passes the world gate - mutation proposes, the world disposes

This is what keeps spice from becoming collapse, and it is what
separates Tessera's loop from the commonLuke demo (where the fitness IS
the world). A mutant is admitted to the archive only if tests/builds/
commits pass and the guard axes do not regress > epsilon. Because
mutation widens the candidate search, it WIDENS the attack surface for
reward hacking (finding 2): strengthen the KernelGuard-style adversarial
checker in proportion. A dieselgate mutant - great on the proxy, fails
in the world - is rejected and becomes a regression test.

### 3.4 Make the schedule adaptive, and use island migration as gene flow

Do not mutate at a constant rate. Trigger mutation BURSTS on stagnation
(no archive improvement for K generations -> reheat) - the "move the
target when the rockets are about to win" moment, turned into a policy.
And since island-GA infrastructure already exists, use occasional
CROSS-ISLAND MIGRATION as a second mutation channel (~1-5% every N
generations) so islands do not each converge on their own local optimum.

### 3.5 Measure it, do not hand-tune it

Treat mutation rate and step distribution as just another axis the
multi-axis eval optimizes, A/B'd via `tessera-ab-harness`, with the
guard axes ensuring "spicier" never means "regressed."

## 4. References

Efficiency:
- Intelligence per Watt - arXiv:2511.07885 (Saad-Falcon, Narayan,
  Akengin et al.); site `intelligence-per-watt.ai`; follow-up "Open
  Jarvis."

Reward hacking / evaluation:
- KernelBench - arXiv:2502.10517 (Ouyang, Guo, Arora, Zhang, Hu, Re,
  Mirhoseini), ICML '25.
- Reward Hacking in Self-Improving Code Agents - under review ICLR '26,
  OpenReview `ikrQWGgxYg`.
- KernelHacks / reward-hack taxonomy and mitigation -
  `makora.com/blog/reward-hacks`. KernelBot / KernelGuard - GPU mode /
  Core Automation (Mark).

Heterogeneous inference:
- Roofline model - Williams, Waterman, Patterson, CACM 2009.
- Marlo heterogeneous-inference talk (YC panel; no single paper).

Kernels:
- ParallelKittens - arXiv:2511.13940 (Sul, Arora, Spector, Re),
  MLSys '26. ThunderKittens - arXiv:2410.20399.

Batch simulation:
- Madrona - "An Extensible, Data-Oriented Architecture for
  High-Performance Many-World Simulation" (Shacklett et al., Stanford).
- Large Batch Simulation for Deep RL - ICLR '21 (Shacklett et al.).

Evolutionary mutation:
- NEAT - Stanley & Miikkulainen, "Evolving Neural Networks through
  Augmenting Topologies," Artificial Life 2002.
- MarI/O - Seth Bling (2015). The Nature of Code - Daniel Shiffman
  (ch. 9). Sentdex neural-network-from-scratch series. Gym Super Mario
  Bros (Python library).
