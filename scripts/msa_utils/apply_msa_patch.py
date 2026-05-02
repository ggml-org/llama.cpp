
#!/usr/bin/env python3
"""
apply_msa_patch.py  (v3 — class-based llama.cpp)
=================================================
Applica il supporto MSA-4B ai sorgenti di llama.cpp con architettura
class-based (make_unique<llm_build_X>), come nelle versioni recenti.

Uso:
    python3 apply_msa_patch.py /path/to/llama.cpp [--dry-run]

Cambiamenti rispetto a v2:
  - Step 0: rimuove il blob MSA inserito male dentro LLM_ARCH_QWEN3VL da v2
  - NON modifica LLM_TENSOR_NAMES (mappa piatta condivisa — ok così)
  - Anchor corretti per la versione class-based
  - llm_build_msa è una sottoclasse di llm_build_qwen3 (eredita il graph)
  - Hparams loader: anchor preciso basato sul testo reale visto nel file
  - Build switch: anchor su KIMI_LINEAR (ultimo caso prima di default)
"""

from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime

DRY_RUN = False


# ─── Utilità ──────────────────────────────────────────────────────────────────

def backup(path: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_suffix(f".bak_{ts}")
    shutil.copy2(path, dst)
    print(f"    Backup → {dst.name}")


def apply_patches(path: Path, patches: list[dict], label: str) -> bool:
    content = path.read_text(encoding="utf-8")
    applied = 0
    failed: list[str] = []

    print(f"\n  ── {path.name}  [{label}]")

    for p in patches:
        desc      = p["desc"]
        anchors   = p["anchors"]
        insertion = p["insertion"]
        check     = p.get("check", insertion[:80])

        if check in content:
            print(f"    SKIP (già presente): {desc}")
            continue

        matched = None
        for anchor in anchors:
            if anchor in content:
                content = content.replace(anchor, insertion + anchor, 1)
                matched = anchor
                applied += 1
                break

        if matched:
            print(f"    OK  : {desc}")
        else:
            print(f"    FAIL: {desc}")
            for i, a in enumerate(anchors, 1):
                print(f"           [{i}] {repr(a[:100])}")
            failed.append(desc)

    if failed:
        print(f"\n    ⚠  {len(failed)} patch non applicate.")

    if applied > 0:
        if not DRY_RUN:
            backup(path)
            path.write_text(content, encoding="utf-8")
            print(f"    Scritto: {path}")
        else:
            print(f"    [DRY-RUN] {applied} modifiche non scritte")
        return True
    return False


def repair_file(path: Path, repairs: list[dict], label: str) -> bool:
    """Rimuove testo malposizionato inserito da patch precedenti."""
    content = path.read_text(encoding="utf-8")
    changed = 0

    print(f"\n  ── {path.name}  [{label}]")

    for r in repairs:
        bad = r["bad_text"]
        if bad in content:
            content = content.replace(bad, "", 1)
            changed += 1
            print(f"    REMOVED : {r['desc']}")
        else:
            print(f"    NOT FOUND: {r['desc']}")

    if changed > 0:
        if not DRY_RUN:
            backup(path)
            path.write_text(content, encoding="utf-8")
            print(f"    Scritto: {path}")
        else:
            print(f"    [DRY-RUN] {changed} rimozioni non scritte")
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# llama-arch.h
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_H_PATCHES = [
    {
        "desc": "enum llm_arch: aggiungi LLM_ARCH_MSA",
        "anchors": ["    LLM_ARCH_UNKNOWN,"],
        "insertion": "    LLM_ARCH_MSA,\n",
        "check": "LLM_ARCH_MSA",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# llama-arch.cpp  — solo la mappa nomi architettura
# LLM_TENSOR_NAMES è piatta e condivisa: NON va toccata
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_CPP_PATCHES = [
    {
        "desc": 'LLM_ARCH_NAMES: aggiungi { LLM_ARCH_MSA, "msa" }',
        "anchors": [
            # Anchor esatto visto nel file: KIMI_LINEAR precede UNKNOWN
            '    { LLM_ARCH_KIMI_LINEAR,      "kimi-linear"      },\n    { LLM_ARCH_UNKNOWN,',
            # Varianti di spaziatura
            '    { LLM_ARCH_KIMI_LINEAR, "kimi-linear" },\n    { LLM_ARCH_UNKNOWN,',
            # Fallback: direttamente su UNKNOWN
            '    { LLM_ARCH_UNKNOWN,          "(unknown)"        },',
            '    { LLM_ARCH_UNKNOWN, "(unknown)" },',
        ],
        "insertion": '    { LLM_ARCH_MSA,            "msa"              },\n',
        "check": '"msa"',
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# llama-model.cpp — Step 0: rimuovi inserzioni sbagliate di v2
# ═══════════════════════════════════════════════════════════════════════════════

# v2 ha inserito il loader case con indentazione a 8 spazi invece di 12,
# e lo ha infilato dentro il blocco QWEN3VL rompendo la sintassi.
V2_BAD_LOADER = (
    "        case LLM_ARCH_MSA:\n"
    "        {\n"
    "            layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd});\n"
    "            layer.ffn_norm  = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM,  \"weight\", i), {n_embd});\n"
    "\n"
    "            layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   \"weight\", i), {n_embd, n_embd_head_k * n_head});\n"
    "            layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   \"weight\", i), {n_embd, n_embd_head_k * n_head_kv});\n"
    "            layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   \"weight\", i), {n_embd, n_embd_head_k * n_head_kv});\n"
    "            layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd});\n"
    "\n"
    "            layer.attn_q_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, \"weight\", i), {n_embd_head_k});\n"
    "            layer.attn_k_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, \"weight\", i), {n_embd_head_k});\n"
    "\n"
    "            layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd, n_ff});\n"
    "            layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd, n_ff});\n"
    "            layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {n_ff, n_embd});\n"
    "\n"
    "            // Router weights MSA \xe2\x80\x94 opzionali (solo layer 18-35)\n"
    "            layer.wq_a = ml.create_tensor(ctx_split,\n"
    "                tn(LLM_TENSOR_ATTN_Q_A, \"weight\", i),\n"
    "                {n_embd, n_embd_head_k * n_head_kv},\n"
    "                llama_model_loader::TENSOR_NOT_REQUIRED);\n"
    "            layer.wk_a = ml.create_tensor(ctx_split,\n"
    "                tn(LLM_TENSOR_ATTN_K_A, \"weight\", i),\n"
    "                {n_embd, n_embd_head_k * n_head_kv},\n"
    "                llama_model_loader::TENSOR_NOT_REQUIRED);\n"
    "        } break;\n"
)

MODEL_CPP_REPAIRS = [
    {"desc": "Loader MSA malposizionato (v2, indentazione 8sp)", "bad_text": V2_BAD_LOADER},
    # build-switch variants from v2
    {
        "desc": "Build-switch MSA (variante cb) da v2",
        "bad_text": (
            "        case LLM_ARCH_MSA:\n"
            "            result = llm_build_msa(ctx0, gf, cb);\n"
            "            break;\n"
        ),
    },
    {
        "desc": "Build-switch MSA (variante res) da v2",
        "bad_text": (
            "        case LLM_ARCH_MSA:\n"
            "            result = llm_build_msa(ctx0, gf, res);\n"
            "            break;\n"
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# llama-model.cpp — hparams loader
# Questo switch deduce il tipo modello dai parametri GGUF.
# Anchor: testo esatto visto nel file con sed.
# ═══════════════════════════════════════════════════════════════════════════════

MSA_HPARAMS_CASE = """\
        case LLM_ARCH_MSA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
"""

HPARAMS_ANCHORS = [
    # Testo esatto visto con sed (36 layer → 4B/8B, poi case 40)
    (
        "        case LLM_ARCH_QWEN3:\n"
        "            {\n"
        "                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);\n"
        "                switch (hparams.n_layer) {\n"
        "                    case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;\n"
        "                    case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;"
    ),
    # Fallback più corto
    (
        "        case LLM_ARCH_QWEN3:\n"
        "            {\n"
        "                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);\n"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# llama-model.cpp — per-layer tensor loader
# Anchor: cerca il case QWEN3 nella funzione di loading tensori
# (diversa dalla funzione hparams — discriminiamo per create_tensor).
# ═══════════════════════════════════════════════════════════════════════════════

MSA_TENSOR_LOADER_CASE = """\
        case LLM_ARCH_MSA:
            {
                layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                layer.ffn_norm  = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd});

                layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
                layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head_kv});
                layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_k * n_head_kv});
                layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

                layer.attn_q_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});
                layer.attn_k_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});

                layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
                layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
                layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});

                // Router weights MSA — opzionali (layer 18-35 only)
                layer.wq_a = ml.create_tensor(ctx_split,
                    tn(LLM_TENSOR_ATTN_Q_A, "weight", i),
                    {n_embd, n_embd_head_k * n_head_kv},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
                layer.wk_a = ml.create_tensor(ctx_split,
                    tn(LLM_TENSOR_ATTN_K_A, "weight", i),
                    {n_embd, n_embd_head_k * n_head_kv},
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            } break;
"""

TENSOR_LOADER_ANCHORS = [
    # The tensor loader QWEN3 case starts with layer.attn_norm (not ml.get_key)
    "        case LLM_ARCH_QWEN3:\n            {\n                layer.attn_norm = ml.create_tensor",
    "        case LLM_ARCH_QWEN3:\n        {\n            layer.attn_norm = ml.create_tensor",
    # Broader fallback: QWEN3MOE follows QWEN3 in the tensor loader
    "        } break;\n        case LLM_ARCH_QWEN3MOE:\n            {\n                layer.attn_norm",
]


# ═══════════════════════════════════════════════════════════════════════════════
# llama-model.cpp — llm_build_msa class
# Sottoclasse di llm_build_qwen3: eredita l'intero graph builder.
# Inseriamo PRIMA della definizione di llm_build_qwen3.
# ═══════════════════════════════════════════════════════════════════════════════

MSA_CLASS = """\
// ─────────────────────────────────────────────────────────────────────────────
// llm_build_msa — MSA-4B (Memory Sparse Attention, arXiv:2603.23516)
//
// Degraded mode: usa GQA standard (Qwen3 backbone) per tutti i 36 layer.
// I pesi router MSA (layer 18-35) sono caricati ma non collegati al graph.
// Il modello funziona correttamente entro il contesto locale KV-cache.
// ─────────────────────────────────────────────────────────────────────────────
struct llm_build_msa : public llm_build_qwen3 {
    llm_build_msa(llama_context & lctx, const llm_build_params & params)
        : llm_build_qwen3(lctx, params) {}
};

"""

# Cerca la definizione della classe qwen3 con vari pattern di ereditarietà
CLASS_ANCHORS = [
    "struct llm_build_qwen3 : public llm_build_base {",
    "struct llm_build_qwen3 : llm_build_base {",
    "struct llm_build_qwen3 : public llm_build_qwen2 {",
    "struct llm_build_qwen3 : public llm_build_qwen {",
    "struct llm_build_qwen3\n{",
    "struct llm_build_qwen3 {",
]


# ═══════════════════════════════════════════════════════════════════════════════
# llama-model.cpp — build switch (make_unique)
# Inseriamo PRIMA di KIMI_LINEAR (ultimo caso nominato prima di default).
# ═══════════════════════════════════════════════════════════════════════════════

MSA_BUILD_SWITCH_CASE = """\
        case LLM_ARCH_MSA:
            {
                llm = std::make_unique<llm_build_msa>(*this, params);
            } break;
"""

BUILD_SWITCH_ANCHORS = [
    # Anchor esatto dal file (testo visto con sed)
    (
        "        case LLM_ARCH_KIMI_LINEAR:\n"
        "            {\n"
        "                llm = std::make_unique<llm_build_kimi_linear>(*this, params);\n"
        "            } break;"
    ),
    # Fallback: inserisci prima di default
    "        default:\n            GGML_ABORT(\"fatal error\");",
    "        default:\n            throw std::runtime_error(\"unknown arch\");",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Build patch list for llama-model.cpp
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_patches(content: str) -> list[dict]:
    # Auto-detect exact class anchor present in this file
    class_anchors = CLASS_ANCHORS[:]
    m = re.search(r'(struct llm_build_qwen3\s*(?::[^{]+)?\{)', content)
    if m:
        found = m.group(1)
        print(f"    Auto-detect classe: {repr(found[:80])}")
        class_anchors.insert(0, found)

    return [
        {
            "desc": "Hparams loader: case LLM_ARCH_MSA",
            "anchors": HPARAMS_ANCHORS,
            "insertion": MSA_HPARAMS_CASE,
            "check": "case LLM_ARCH_MSA:\n            {\n                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS",
        },
        {
            "desc": "Tensor loader: case LLM_ARCH_MSA",
            "anchors": TENSOR_LOADER_ANCHORS,
            "insertion": MSA_TENSOR_LOADER_CASE,
            "check": "case LLM_ARCH_MSA:\n            {\n                layer.attn_norm",
        },
        {
            "desc": "Classe llm_build_msa (prima di llm_build_qwen3)",
            "anchors": class_anchors,
            "insertion": MSA_CLASS,
            "check": "struct llm_build_msa",
        },
        {
            "desc": "Build switch: make_unique<llm_build_msa>",
            "anchors": BUILD_SWITCH_ANCHORS,
            "insertion": MSA_BUILD_SWITCH_CASE,
            "check": "llm_build_msa>(*this, params)",
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Manual instructions
# ═══════════════════════════════════════════════════════════════════════════════

def print_manual_instructions(llama_dir: Path) -> None:
    src = llama_dir / "src"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║      Istruzioni manuali per le patch non applicate               ║
╚══════════════════════════════════════════════════════════════════╝

Grep utili per trovare le righe esatte nella tua versione:

  grep -n 'LLM_ARCH_UNKNOWN'   {src}/llama-arch.h
  grep -n 'LLM_ARCH_UNKNOWN\\|KIMI_LINEAR'  {src}/llama-arch.cpp
  grep -n 'struct llm_build_qwen3' {src}/llama-model.cpp
  grep -n 'LLM_ARCH_QWEN3[^A-Z]'  {src}/llama-model.cpp | head -20
  grep -n 'KIMI_LINEAR\\|default:' {src}/llama-model.cpp | tail -20

Modifiche da fare:

1. llama-arch.h  — enum llm_arch, PRIMA di LLM_ARCH_UNKNOWN:
       LLM_ARCH_MSA,

2. llama-arch.cpp — mappa nomi, PRIMA di LLM_ARCH_UNKNOWN:
       {{ LLM_ARCH_MSA, "msa" }},

3. llama-model.cpp — hparams loader switch, PRIMA di case LLM_ARCH_QWEN3:
       case LLM_ARCH_MSA:
           {{ ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
             switch (hparams.n_layer) {{
               case 36: type = hparams.n_embd==2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
               default: type = LLM_TYPE_UNKNOWN;
             }}
           }} break;

4. llama-model.cpp — tensor loader switch, PRIMA di case LLM_ARCH_QWEN3:
   (vedi MSA_TENSOR_LOADER_CASE nel sorgente di questo script)

5. llama-model.cpp — PRIMA di struct llm_build_qwen3:
       struct llm_build_msa : public llm_build_qwen3 {{
           llm_build_msa(llama_context & lctx, const llm_build_params & params)
               : llm_build_qwen3(lctx, params) {{}}
       }};

6. llama-model.cpp — build switch, PRIMA di case LLM_ARCH_KIMI_LINEAR:
       case LLM_ARCH_MSA:
           {{ llm = std::make_unique<llm_build_msa>(*this, params); }} break;
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global DRY_RUN
    ap = argparse.ArgumentParser(description="MSA-4B patch v3 (class-based llama.cpp)")
    ap.add_argument("llama_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    llama_dir = args.llama_dir.resolve()
    DRY_RUN   = args.dry_run

    if not llama_dir.is_dir():
        sys.exit(f"Directory non trovata: {llama_dir}")

    src = llama_dir / "src"
    arch_h    = src / "llama-arch.h"
    arch_cpp  = src / "llama-arch.cpp"
    model_cpp = src / "llama-model.cpp"

    for f in [arch_h, arch_cpp, model_cpp]:
        if not f.exists():
            sys.exit(f"File non trovato: {f}")

    tag = "[DRY-RUN] " if DRY_RUN else ""
    print(f"\n{tag}MSA-4B patch applicator v3  (class-based llama.cpp)")
    print(f"  {llama_dir}\n")

    # ── Step 0: undo v2 damage ────────────────────────────────────────────────
    print("Step 0: Repair — rimozione inserzioni malposte da v2...")
    repair_file(model_cpp, MODEL_CPP_REPAIRS, "undo v2")

    # ── Step 1: llama-arch.h ──────────────────────────────────────────────────
    print("\nStep 1: llama-arch.h")
    apply_patches(arch_h, ARCH_H_PATCHES, "enum llm_arch")

    # ── Step 2: llama-arch.cpp ────────────────────────────────────────────────
    print("\nStep 2: llama-arch.cpp")
    apply_patches(arch_cpp, ARCH_CPP_PATCHES, "arch name map")

    # ── Step 3: llama-model.cpp ───────────────────────────────────────────────
    print("\nStep 3: llama-model.cpp")
    content = model_cpp.read_text(encoding="utf-8")
    model_patches = build_model_patches(content)
    apply_patches(model_cpp, model_patches,
                  "hparams loader + tensor loader + class + build switch")

    print_manual_instructions(llama_dir)

    if not DRY_RUN:
        print(f"""{'='*60}
Ricompila:
  cd {llama_dir}
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)

Testa:
  ./build/bin/llama-cli -m /path/to/MSA-4B-F16.gguf -p "Ciao!" -n 64
{'='*60}
""")


if __name__ == "__main__":
    main()