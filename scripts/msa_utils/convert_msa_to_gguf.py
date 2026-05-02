#!/usr/bin/env python3
"""
convert_msa_to_gguf.py
======================
Converte EverMind-AI/MSA-4B (HuggingFace safetensors) in formato GGUF
compatibile con llama.cpp patchato con msa.cpp / apply_msa_patch.py.

Uso:
    python convert_msa_to_gguf.py <model_dir> [--outfile output.gguf] [--outtype f16|bf16|f32]
    python convert_msa_to_gguf.py <model_dir> --list-tensors   # ispeziona nomi HF

Requisiti:
    pip install gguf safetensors torch

Architettura MSA-4B (da paper arXiv:2603.23516):
    - 36 layer totali, Qwen3-4B backbone (hidden=2560, Q=32h, KV=8h, SwiGLU)
    - layer  0-17: standard Qwen3 GQA attention
    - layer 18-35: Memory Sparse Attention
                   router_q_proj.0.weight  shape=(4096, 2560)  → blk.N.attn_router_q.weight
                   router_k_proj.0.weight  shape=(1024, 2560)  → blk.N.attn_router_k.weight
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


# ─── Architettura GGUF ───────────────────────────────────────────────────────
MSA_ARCH_NAME = "msa"   # deve corrispondere a llama-arch.cpp: { LLM_ARCH_MSA, "msa" }

# Chiavi metadati custom MSA (lette da llama.cpp se presenti, ignorate altrimenti)
KEY_MSA_FIRST_MSA_LAYER = "msa.attention.first_msa_layer"
KEY_MSA_NUM_MSA_LAYERS  = "msa.attention.num_msa_layers"
KEY_MSA_ROUTER_TOP_K    = "msa.attention.router_top_k"

# Tensori da saltare completamente.
# I router weights MSA (attn_router_q/k) NON vengono scritti nel GGUF:
# llama.cpp conta i tensori attesi dal loader e quelli presenti nel file —
# se ce ne sono di extra che il loader non registra, lancia
# "wrong number of tensors". Poiché llm_build_msa non usa i router nel
# graph (degraded mode), è più semplice non includerli affatto.
SKIP_TENSOR_NAMES = {
    "temperature",          # scalare float32, non è un peso di rete
}

# Suffissi dei router MSA da escludere (layer 18-35)
SKIP_TENSOR_SUFFIXES = (
    ".router_q_proj.0.weight",
    ".router_k_proj.0.weight",
    ".router_q_proj.weight",
    ".router_k_proj.weight",
    ".msa_q_proj.weight",
    ".msa_k_proj.weight",
    ".indexer_q.weight",
    ".indexer_k.weight",
)


# ─── Mappatura nomi tensori HF → GGUF ────────────────────────────────────────

def build_tensor_map(num_layers: int, first_msa_layer: int) -> dict[str, str]:
    """Restituisce {nome_hf: nome_gguf} per tutti i tensori del modello."""
    m: dict[str, str] = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight":         "output_norm.weight",
        "lm_head.weight":            "output.weight",
    }

    for i in range(num_layers):
        b   = f"blk.{i}"
        p   = f"model.layers.{i}"
        s   = f"{p}.self_attn"
        mlp = f"{p}.mlp"

        # Norme
        m[f"{p}.input_layernorm.weight"]         = f"{b}.attn_norm.weight"
        m[f"{p}.post_attention_layernorm.weight"] = f"{b}.ffn_norm.weight"

        # Attenzione QKV + output (tutti i layer)
        m[f"{s}.q_proj.weight"] = f"{b}.attn_q.weight"
        m[f"{s}.k_proj.weight"] = f"{b}.attn_k.weight"
        m[f"{s}.v_proj.weight"] = f"{b}.attn_v.weight"
        m[f"{s}.o_proj.weight"] = f"{b}.attn_output.weight"

        # QK-norm (Qwen3 style)
        m[f"{s}.q_norm.weight"] = f"{b}.attn_q_norm.weight"
        m[f"{s}.k_norm.weight"] = f"{b}.attn_k_norm.weight"

        # MLP SwiGLU
        m[f"{mlp}.gate_proj.weight"] = f"{b}.ffn_gate.weight"
        m[f"{mlp}.up_proj.weight"]   = f"{b}.ffn_up.weight"
        m[f"{mlp}.down_proj.weight"] = f"{b}.ffn_down.weight"

        # Router MSA (layer 18-35).
        # Il modello reale usa il nome "router_{q,k}_proj.0.weight"
        # (lista di proiezioni, indice 0). Mappiamo tutte le varianti plausibili
        # sullo stesso nome GGUF così uno qualsiasi matcha.
        if i >= first_msa_layer:
            # Nome reale confermato dall'output del modello:
            m[f"{s}.router_q_proj.0.weight"] = f"{b}.attn_router_q.weight"
            m[f"{s}.router_k_proj.0.weight"] = f"{b}.attn_router_k.weight"
            # Fallback senza indice (nel caso future versioni lo tolgano):
            m[f"{s}.router_q_proj.weight"]   = f"{b}.attn_router_q.weight"
            m[f"{s}.router_k_proj.weight"]   = f"{b}.attn_router_k.weight"
            # Altri alias visti in fork / paper:
            m[f"{s}.msa_q_proj.weight"]      = f"{b}.attn_router_q.weight"
            m[f"{s}.msa_k_proj.weight"]      = f"{b}.attn_router_k.weight"
            m[f"{s}.indexer_q.weight"]       = f"{b}.attn_router_q.weight"
            m[f"{s}.indexer_k.weight"]       = f"{b}.attn_router_k.weight"

    return m


# ─── Caricamento safetensors ──────────────────────────────────────────────────

def load_safetensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Carica tutti i tensori dai file .safetensors.

    Usa framework='pt' perché numpy non supporta bfloat16.
    I tensori bf16 vengono convertiti in float32 prima di tornare numpy.
    """
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            idx = json.load(f)
        shard_files = sorted(set(idx["weight_map"].values()))
    else:
        shard_files = sorted(model_dir.glob("*.safetensors"))
        if not shard_files:
            sys.exit(f"Nessun file .safetensors trovato in {model_dir}")

    tensors: dict[str, np.ndarray] = {}
    for fname in shard_files:
        path = model_dir / fname if isinstance(fname, str) else fname
        print(f"  Carico {path.name}...")
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if t.dtype == torch.bfloat16:
                    t = t.to(torch.float32)
                tensors[key] = t.numpy()
    return tensors


# ─── Conversione dtype ────────────────────────────────────────────────────────

def to_gguf_dtype(
    arr: np.ndarray, outtype: str, name: str = ""
) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    # 1D tensors (norms, biases) must always be F32.
    # ggml_rms_norm output is F32; ggml_mul requires both operands same type.
    # Saving norm weights as F16 causes "unsupported types: f32 * f16" at runtime.
    if arr.ndim == 1:
        return arr.astype(np.float32), gguf.GGMLQuantizationType.F32

    if outtype == "f32":
        return arr.astype(np.float32), gguf.GGMLQuantizationType.F32
    # f16 o bf16: GGUF salva come float16 numpy
    if arr.dtype == np.float16:
        return arr, gguf.GGMLQuantizationType.F16
    return arr.astype(np.float32).astype(np.float16), gguf.GGMLQuantizationType.F16


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def write_tokenizer(writer: gguf.GGUFWriter, model_dir: Path, config: dict) -> None:
    vocab_size       = config.get("vocab_size", 151936)
    tok_json         = model_dir / "tokenizer.json"
    vocab_json       = model_dir / "vocab.json"
    tokenizer_config = model_dir / "tokenizer_config.json"

    if not tok_json.exists():
        print("  WARN: tokenizer.json non trovato — tokenizer non scritto nel GGUF")
        return

    with open(tok_json) as f:
        tok = json.load(f)

    # Vocab: dict str→id
    vocab = tok.get("model", {}).get("vocab", {})
    if not vocab and vocab_json.exists():
        with open(vocab_json) as f:
            vocab = json.load(f)

    id_to_token = {v: k for k, v in vocab.items()}

    # Aggiungi gli added_tokens (token speciali: <|im_start|>, <|im_end|>, ecc.)
    # Questi hanno ID >= 151643 e sono necessari per il chat template.
    for at in tok.get("added_tokens", []):
        at_id      = at["id"]
        at_content = at["content"]
        id_to_token[at_id] = at_content

    # Merges: normalizza sia "tok1 tok2" (str) che ["tok1","tok2"] (lista)
    raw_merges = tok.get("model", {}).get("merges", [])
    merges: list[str] = []
    for entry in raw_merges:
        if isinstance(entry, (list, tuple)):
            merges.append(" ".join(str(x) for x in entry))
        else:
            merges.append(str(entry))

    tokens:      list[bytes] = []
    scores:      list[float] = []
    token_types: list[int]   = []

    # Token type codes: 1=NORMAL, 3=CONTROL (special tokens)
    special_ids = {at["id"] for at in tok.get("added_tokens", []) if at.get("special", False)}

    for i in range(vocab_size):
        tok_str = id_to_token.get(i, f"[PAD{i}]")
        tokens.append(tok_str.encode("utf-8") if isinstance(tok_str, str) else tok_str)
        scores.append(0.0)
        token_types.append(3 if i in special_ids else 1)

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    # Token speciali
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)
        bos = tc.get("bos_token")
        eos = tc.get("eos_token")
        if bos and bos in vocab:
            writer.add_bos_token_id(vocab[bos])
        if eos and eos in vocab:
            writer.add_eos_token_id(vocab[eos])

    if merges:
        # add_token_merges() può produrre tipo GGUF 9 (ARRAY-of-ARRAY) su alcune
        # versioni della libreria — scriviamo la chiave direttamente come STRING_ARRAY.
        writer.add_array("tokenizer.ggml.merges", merges)

    print(f"  Tokenizer: gpt2, vocab size={vocab_size}, merges={len(merges)}")


# ─── Converter principale ─────────────────────────────────────────────────────

def convert(model_dir: Path, outfile: Path, outtype: str) -> None:
    print(f"\n=== MSA-4B → GGUF Converter ===")
    print(f"  Modello : {model_dir}")
    print(f"  Output  : {outfile}")
    print(f"  Tipo    : {outtype.upper()}")
    print()

    # 1. config.json
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"config.json non trovato in {model_dir}")
    with open(cfg_path) as f:
        config = json.load(f)

    # Parametri architettura (default Qwen3-4B se assenti)
    num_layers   = config.get("num_hidden_layers", 36)
    hidden_size  = config.get("hidden_size", 2560)
    ff_size      = config.get("intermediate_size", 9728)
    num_heads    = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", 8)
    rms_eps      = config.get("rms_norm_eps", 1e-6)
    rope_theta   = config.get("rope_theta", 1_000_000.0)
    vocab_size   = config.get("vocab_size", 151936)
    ctx_len      = config.get("max_position_embeddings", 32768)
    head_dim     = config.get("head_dim", hidden_size // num_heads)

    # Parametri MSA custom
    first_msa_layer = config.get("first_msa_layer",
                      config.get("num_msa_start_layer", num_layers // 2))
    num_msa_layers  = config.get("num_msa_layers",
                      config.get("msa_layer_count", num_layers - first_msa_layer))
    router_top_k    = config.get("msa_router_top_k",
                      config.get("router_top_k", 5))

    print(f"  Architettura:")
    print(f"    Layers        : {num_layers}  (MSA da layer {first_msa_layer}, n={num_msa_layers})")
    print(f"    Hidden size   : {hidden_size}  head_dim={head_dim}")
    print(f"    Q/KV heads    : {num_heads}/{num_kv_heads}")
    print(f"    RoPE theta    : {rope_theta}")
    print(f"    Router top-k  : {router_top_k}")
    print()

    # 2. Mappa tensori
    tensor_map = build_tensor_map(num_layers, first_msa_layer)

    # 3. Carica pesi
    print("Caricamento safetensors...")
    hf_tensors = load_safetensors(model_dir)
    print(f"  {len(hf_tensors)} tensori caricati")
    print()

    # Report tensori non mappati
    unknown = [
        k for k in hf_tensors
        if k not in tensor_map and k not in SKIP_TENSOR_NAMES
    ]
    if unknown:
        print(f"  ⚠ Tensori HF senza mappatura GGUF ({len(unknown)}):")
        for k in sorted(unknown):
            print(f"    {k}  shape={hf_tensors[k].shape}  dtype={hf_tensors[k].dtype}")
        print()
    else:
        print("  ✓ Tutti i tensori mappati correttamente")
        print()

    # 4. GGUFWriter
    writer = gguf.GGUFWriter(str(outfile), MSA_ARCH_NAME)

    # 5. Metadati generali
    writer.add_name("MSA-4B")
    writer.add_description(
        "EverMind MSA-4B — Memory Sparse Attention, Qwen3 backbone, "
        "100M-token memory context (degraded: local KV only)"
    )
    writer.add_source_url("https://huggingface.co/EverMind-AI/MSA-4B")

    # 6. Iperparametri architettura
    writer.add_block_count(num_layers)
    writer.add_context_length(ctx_len)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(ff_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_layer_norm_rms_eps(rms_eps)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(rope_theta)
    writer.add_vocab_size(vocab_size)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)  # required: without this n_embd_head_v_full = n_embd/n_head = 80 (wrong)

    # Metadati MSA custom
    writer.add_uint32(KEY_MSA_FIRST_MSA_LAYER, first_msa_layer)
    writer.add_uint32(KEY_MSA_NUM_MSA_LAYERS,  num_msa_layers)
    writer.add_uint32(KEY_MSA_ROUTER_TOP_K,    router_top_k)

    # File type
    ftype_map = {
        "f32":  gguf.LlamaFileType.ALL_F32,
        "f16":  gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    }
    writer.add_file_type(ftype_map.get(outtype, gguf.LlamaFileType.MOSTLY_F16))

    # 7. Tokenizer
    print("Scrittura tokenizer...")
    write_tokenizer(writer, model_dir, config)
    print()

    # 8. Tensori
    print("Scrittura tensori...")
    written  = 0
    skipped  = 0
    unmapped = 0

    for hf_name, arr in hf_tensors.items():
        # Salta scalari e tensori speciali
        if hf_name in SKIP_TENSOR_NAMES:
            skipped += 1
            continue

        # Salta i router weights MSA: non sono registrati dal tensor loader
        # (llm_build_msa degraded mode non li usa) e causerebbero
        # "wrong number of tensors" al caricamento.
        if any(hf_name.endswith(sfx) for sfx in SKIP_TENSOR_SUFFIXES):
            skipped += 1
            continue

        # Salta tensori stringa / object
        if arr.dtype == object or arr.dtype.kind == 'U':
            skipped += 1
            continue

        gguf_name = tensor_map.get(hf_name)
        if gguf_name is None:
            # Tensore non mappato: prefissa con "hf." e includi comunque
            gguf_name = "hf." + hf_name
            unmapped += 1

        out_arr, dtype_enum = to_gguf_dtype(arr, outtype, hf_name)
        writer.add_tensor(gguf_name, out_arr, raw_dtype=dtype_enum)
        written += 1

        if written % 50 == 0:
            print(f"  {written}/{len(hf_tensors)} tensori scritti...")

    print(f"  {written} tensori scritti")
    if skipped:
        print(f"  {skipped} tensori saltati (scalari / stringa)")
    if unmapped:
        print(f"  {unmapped} tensori salvati con prefisso 'hf.' (non mappati)")
    print()

    # 9. Scrivi su disco
    print(f"Scrittura GGUF su {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"\n✓ GGUF scritto: {outfile}  ({size_mb:.1f} MB)")
    print(f"""
Prossimi passi:

  1. Applica il patch C++ a llama.cpp (se non già fatto):
       python3 apply_msa_patch.py /path/to/llama.cpp
       cp msa.cpp /path/to/llama.cpp/src/models/
       cp models.h /path/to/llama.cpp/src/models/

  2. Ricompila:
       cd /path/to/llama.cpp
       cmake --build build -j$(nproc)

  3. Testa:
       ./build/bin/llama-cli -m {outfile} -p 'Ciao!' -n 128

  4. Quantizza (opzionale, risparmia VRAM):
       ./build/bin/llama-quantize {outfile} \\
           {outfile.with_name(outfile.stem + '-Q4_K_M.gguf')} Q4_K_M
""")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MSA-4B HuggingFace safetensors → GGUF converter"
    )
    ap.add_argument("model_dir", type=Path,
                    help="Cartella del modello HF scaricato")
    ap.add_argument("--outfile", type=Path, default=None,
                    help="File GGUF di output (default: <model_dir>/MSA-4B-<TYPE>.gguf)")
    ap.add_argument("--outtype", choices=["f16", "bf16", "f32"], default="f16",
                    help="Tipo output (default: f16)")
    ap.add_argument("--list-tensors", action="store_true",
                    help="Stampa tutti i nomi dei tensori HF ed esci")
    args = ap.parse_args()

    model_dir: Path = args.model_dir.resolve()
    if not model_dir.is_dir():
        sys.exit(f"Directory non trovata: {model_dir}")

    if args.list_tensors:
        print("Caricamento tensori per ispezione...")
        tensors = load_safetensors(model_dir)
        print(f"{'Nome HF':<72} {'Shape':<25} {'Dtype'}")
        print("-" * 110)
        for name, arr in sorted(tensors.items()):
            print(f"  {name:<70}  {str(arr.shape):<25}  {arr.dtype}")
        return

    outfile = args.outfile or (model_dir / f"MSA-4B-{args.outtype.upper()}.gguf")
    convert(model_dir, outfile, args.outtype)


if __name__ == "__main__":
    main()