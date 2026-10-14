import json

categories = {
    "code": [
        ("python", "Red-Black Tree implementation with recoloring, rotations, deletion fixups, and property verifications."),
        ("python", "Asynchronous web crawler with asyncio, aiohttp, priority queue, per-domain rate limiting, and backpressure."),
        ("python", "Bytecode compiler and stack-based virtual machine executing arithmetic, jumps, and function calls."),
        ("python", "Custom PyTorch autograd engine implementing Tensor, backward DAG traversal, and Adam optimizer from scratch."),
        ("rust", "Work-stealing thread pool with Chase-Lev deques, crossbeam-style task stealing, and graceful atomic shutdown."),
        ("rust", "Zero-copy HTTP/1.1 streaming request parser operating on raw byte slices with chunked transfer state machine."),
        ("rust", "Lock-free concurrent skip list with atomic compare-and-swap pointer tagging and hazard pointers."),
        ("rust", "B-Tree implementation with node splitting, merging, rebalancing, and binary search within nodes."),
        ("c", "Buddy memory allocator with order 0 (4KB) to order 10 (4MB) blocks and integrated slab caches."),
        ("c", "High-performance Linux network server using edge-triggered epoll, ring buffers, signalfd, and timerfd."),
        ("c", "Userspace cooperative green thread scheduler using swapcontext, makecontext, and sleep queues."),
        ("c", "Custom dynamic memory profiler hooking malloc, calloc, realloc, free using LD_PRELOAD and backtrace."),
        ("sql", "PostgreSQL recursive CTE computing 10-level corporate org tree depth, cumulative budgets, and rankings."),
        ("sql", "E-commerce customer lifetime value, cohort retention matrix, and churn risk indicators using window functions."),
        ("sql", "Financial transaction fraud detection pipeline using rolling partition windows and standard deviation filters."),
        ("sql", "Inventory replenishment reorder point optimizer with lead-time percentiles and safety stock calculations."),
        ("go", "Distributed key-value store with Raft consensus, log compaction, leader election, and snapshotting."),
        ("go", "Concurrent bounded rate limiter using token bucket algorithm with sync/atomic and select channels."),
        ("typescript", "Complete type-safe GraphQL query parser and AST validator with custom schema directives."),
        ("typescript", "Reactive state management library implementing signals, computed values, effects, and batching."),
        ("cpp", "SIMD-vectorized matrix multiplication library using AVX-512 intrinsics with cache tiling."),
        ("cpp", "Lock-free MPMC ring buffer queue with cacheline padding and memory order acquire/release."),
        ("kernel", "Linux kernel character device driver with ioctl handlers, mutex locking, and sysfs attributes."),
        ("compiler", "LR(1) parser generator constructing canonical collections of items, action/goto tables, and AST generation."),
        ("python", "Distributed map-reduce engine with master coordinator, RPC worker tasks, and shuffle sorting."),
        ("rust", "Raft consensus implementation handling AppendEntries RPCs, vote requests, and persistent state."),
        ("c", "Custom ELF binary parser reading program headers, section headers, symbol tables, and relocations."),
        ("sql", "Temporal database schema tracking bi-temporal valid-time and transaction-time tables with audit trails."),
        ("python", "Graph neural network message passing framework implementing GCN and GAT convolution layers."),
        ("rust", "Asynchronous TLS 1.3 handshake state machine parsing ClientHello, ServerHello, and certificate chains."),
        ("c", "Ext2 filesystem inspection tool reading superblock, block group descriptors, and inode tables."),
        ("typescript", "Virtual DOM diffing and reconciliation algorithm computing minimum tree edit patches.")
    ],
    "math": [
        ("linear_algebra", "Rigorous step-by-step proof of the Spectral Theorem for real symmetric matrices with mathematical induction."),
        ("pde", "Analytical solution of the 1D heat equation with homogeneous Dirichlet boundary conditions via separation of variables."),
        ("game_theory", "Infinitely repeated Prisoner's Dilemma analysis proving Grim Trigger equilibrium via One-Shot Deviation Principle."),
        ("probability", "Gambler's Ruin recurrence relation derivation, characteristic roots, and Martingale stopping time analysis."),
        ("abstract_algebra", "Proof of the First Isomorphism Theorem for groups and classification of finite abelian groups."),
        ("topology", "Proof of Tychonoff's Theorem for arbitrary products of compact topological spaces using ultrafilters."),
        ("complex_analysis", "Derivation of Cauchy's Integral Formula and evaluation of contour integrals using Residue Theorem."),
        ("calculus_variations", "Derivation of the Euler-Lagrange equations from Hamilton's principle of stationary action with constraints."),
        ("differential_geometry", "Computation of Riemann curvature tensor, Ricci tensor, and Ricci scalar on a Riemannian 2-sphere."),
        ("number_theory", "Proof of the Quadratic Reciprocity Law using Gauss sums and Legendre symbols."),
        ("real_analysis", "Rigorous proof of the Stone-Weierstrass Approximation Theorem on compact metric spaces."),
        ("combinatorics", "Derivation of the master theorem for divide-and-conquer recurrences with tight asymptotic bounds."),
        ("optimization", "Karush-Kuhn-Tucker (KKT) necessary and sufficient conditions derivation for convex optimization with duality."),
        ("stochastic", "Itô's Lemma derivation and analytical solution to the Black-Scholes partial differential equation."),
        ("graph_theory", "Proof of the Max-Flow Min-Cut theorem using residual networks and augmenting paths."),
        ("numerical", "Convergence rate proof for the Conjugate Gradient method on symmetric positive definite linear systems.")
    ],
    "chat": [
        ("physics", "Detailed quantum entanglement and CHSH inequality derivation proving Tsirelson's bound vs classical limits."),
        ("physics", "Derivation of the Schwarzschild metric from Einstein's vacuum field equations with Newtonian limit."),
        ("biology", "Molecular biology of CRISPR-Cas9 genome editing, PAM recognition, R-loop formation, and repair mechanisms."),
        ("history", "Analytical comparison of industrial modernization in Meiji Japan vs 19th-century Imperial Prussia."),
        ("astrophysics", "Thermodynamics and hydrodynamics of stellar nucleosynthesis from hydrogen burning to iron core collapse."),
        ("neuroscience", "Biophysics of action potential generation: Hodgkin-Huxley model, ion channel gating kinetics, and Nernst potentials."),
        ("chemistry", "Transition state theory and Marcus electron transfer theory in outer-sphere redox reactions."),
        ("philosophy", "Comparative essay on Kantian deontological ethics vs Scanlonian contractualism on moral duties."),
        ("economics", "Monetary policy transmission mechanisms under zero lower bound: quantitative easing vs forward guidance."),
        ("quantum_computing", "Detailed mathematical operation of Shor's quantum factoring algorithm and quantum Fourier transform."),
        ("materials", "Electronic band structure of graphene and topological insulators: Dirac cones and Berry phase."),
        ("linguistics", "Generative syntax and Minimalist Program: Merge operation, Phase Theory, and feature valuation."),
        ("climate", "Geophysical fluid dynamics of oceanic thermohaline circulation and atmospheric Rossby waves."),
        ("evolutionary_bio", "Kin selection, Hamilton's rule, and evolutionary game theory in the evolution of eusociality."),
        ("history", "The Fall of the Western Roman Empire: institutional decay, economic crises, and barbarian federates."),
        ("epistemology", "The Gettier problem and contemporary responses: reliabilism, virtue epistemology, and sensitivity conditions.")
    ],
    "tool": [
        ("cloud", "Comprehensive Terraform-like JSON specification provisioning a multi-region HA Kubernetes cluster on AWS."),
        ("agent", "International airline customer support agent router returning strictly formatted structured JSON actions."),
        ("medical", "Emergency department clinical triaging assistant outputting structured JSON with ICD-10 codes and ESI score."),
        ("finance", "High-frequency trading algorithmic execution plan using Almgren-Chriss model as strict JSON schema."),
        ("devops", "CI/CD multi-stage deployment pipeline definition JSON with canary stages, rollback policies, and metrics."),
        ("database", "Zero-downtime database schema migration plan JSON with table renaming, shadow columns, and backfills."),
        ("security", "SOC2 compliance audit evidence payload JSON mapping security controls to AWS IAM policies and logs."),
        ("ecommerce", "Omnichannel order fulfillment dispatch optimization plan JSON assigning orders to nearest fulfillment hubs."),
        ("billing", "Usage-based metered billing calculation schema JSON handling tiered rates, prepaid credits, and overages."),
        ("iot", "Industrial telemetry anomaly detection event JSON schema for smart grid sensor networks with alert thresholds."),
        ("nlp", "Named entity recognition and relation extraction payload JSON for biomedical clinical trial publications."),
        ("travel", "Dynamic multi-modal route navigation itinerary JSON combining flights, trains, and rideshares with layovers."),
        ("insurance", "Automated motor vehicle insurance claim assessment report JSON with damage estimates and fraud scores."),
        ("hr", "Global payroll tax withholding and compliance calculation JSON for multi-jurisdiction distributed employees."),
        ("gaming", "Massively multiplayer online game matchmaking lobby state JSON with ELO rankings and latency brackets."),
        ("logistics", "Global container shipping freight route dispatch JSON with port customs clearance milestones and ETA predictions.")
    ]
}

prompts = []
pid = 0
for cat, items in categories.items():
    for dom, desc in items:
        prompts.append({
            "id": pid,
            "category": cat,
            "domain": dom,
            "prompt": desc
        })
        pid += 1

with open("prompts/dataset_80.jsonl", "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")

print(f"Generated {len(prompts)} prompts in prompts/dataset_80.jsonl")
