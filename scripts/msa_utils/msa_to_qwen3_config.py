#!/usr/bin/env python3
"""
msa_to_qwen3_config.py
======================
Patcha il config.json di EverMind-AI/MSA-4B per renderlo compatibile
con llama.cpp come architettura Qwen3 standard — senza nessun patch C++.

I layer MSA-specifici vengono ignorati (router Q/K non collegati).
Il backbone Qwen3 funziona completamente: ottieni un modello 4B
a piena qualità entro il contesto locale.

Uso:
    python3 msa_to_qwen3_config.py <model_dir> [--no-convert]

    --no-convert  Solo patcha config.json, non lancia la conversione GGUF

Requisiti:
    pip install gguf safetensors transformers sentencepiece
    git clone https://github.com/ggml-org/llama.cpp  (per convert_hf_to_gguf.py)
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ─── Config Qwen3-4B canonica ────────────────────────────────────────────────
# Tutti i campi che llama.cpp si aspetta per arch=qwen3.
# I valori sono i default Qwen3-4B-Instruct; verranno sovrascritti dai
# valori reali trovati nel config.json del modello se presenti.

QWEN3_DEFAULTS = {
    "architectures": ["Gemma2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "max_position_embeddings": 32768,
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_k_v_heads": 8,
    "num_heads": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "vocab_size": 151936,
    "head_dim": 128,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "sliding_window": None,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
}

# Campi MSA custom da rimuovere (non capiti da llama.cpp/transformers standard)
MSA_ONLY_FIELDS = {
    "msa_layer_count",
    "num_msa_layers",
    "first_msa_layer",
    "num_msa_start_layer",
    "msa_router_top_k",
    "router_top_k",
    "msa_chunk_size",
    "msa_top_k",
    "memory_type",
    "sparse_attn_type",
    "doc_rope_type",
}


def patch_config(model_dir: Path, dry_run: bool = False) -> Path:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"config.json non trovato in {model_dir}")

    with open(cfg_path) as f:
        original = json.load(f)

    print("Config originale:")
    print(f"  model_type      : {original.get('model_type', 'N/A')}")
    print(f"  architectures   : {original.get('architectures', 'N/A')}")
    print(f"  hidden_size     : {original.get('hidden_size', 'N/A')}")
    print(f"  num_layers      : {original.get('num_hidden_layers', 'N/A')}")
    print(f"  num_heads       : {original.get('num_attention_heads', 'N/A')}")
    print(f"  num_kv_heads    : {original.get('num_key_value_heads', 'N/A')}")
    print(f"  rope_theta      : {original.get('rope_theta', 'N/A')}")
    print(f"  vocab_size      : {original.get('vocab_size', 'N/A')}")
    msa_fields_found = [k for k in original if k in MSA_ONLY_FIELDS]
    if msa_fields_found:
        print(f"  Campi MSA custom: {msa_fields_found}")
    print()

    # Parti dai default Qwen3
    patched = dict(QWEN3_DEFAULTS)

    # Sovrascrivi con i valori reali del modello (tutto tranne i campi MSA)
    for k, v in original.items():
        if k not in MSA_ONLY_FIELDS:
            patched[k] = v

    # Forza i campi critici per la compatibilità
    patched["model_type"] = "qwen3"
    patched["architectures"] = ["Qwen3ForCausalLM"]

    # head_dim esplicito (llama.cpp lo usa)
    hidden_size = patched.get("hidden_size", 2560)
    num_heads = patched.get("num_attention_heads", 32)
    patched["head_dim"] = hidden_size // num_heads

    # Rimuovi campi MSA
    removed = []
    for field in MSA_ONLY_FIELDS:
        if field in patched:
            del patched[field]
            removed.append(field)

    print("Config patchato:")
    print(f"  model_type      : {patched['model_type']}")
    print(f"  architectures   : {patched['architectures']}")
    print(f"  head_dim        : {patched['head_dim']}")
    if removed:
        print(f"  Rimossi         : {removed}")
    print()

    if dry_run:
        print("[DRY-RUN] config.json non modificato")
        return cfg_path

    # Backup del config originale
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = model_dir / f"config.json.msa_backup_{ts}"
    shutil.copy2(cfg_path, bak)
    print(f"Backup originale: {bak.name}")

    # Scrivi config patchato
    with open(cfg_path, "w") as f:
        json.dump(patched, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Config scritto  : {cfg_path}")
    return cfg_path


def find_convert_script(llama_dir: Path | None) -> Path | None:
    """Cerca convert_hf_to_gguf.py in llama.cpp o nel PATH."""
    candidates = []

    if llama_dir:
        candidates += [
            llama_dir / "convert_hf_to_gguf.py",
            llama_dir / "convert-hf-to-gguf.py",
        ]

    # cerca nella directory corrente e nelle sottocartelle comuni
    candidates += [
        Path("convert_hf_to_gguf.py"),
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path("../llama.cpp/convert_hf_to_gguf.py"),
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def run_conversion(
    model_dir: Path, outfile: Path, outtype: str, llama_dir: Path | None
) -> None:
    script = find_convert_script(llama_dir)

    if script is None:
        print("⚠ convert_hf_to_gguf.py non trovato.")
        print("  Clona llama.cpp e passa --llama-dir:")
        print("    git clone https://github.com/ggml-org/llama.cpp")
        print(f"    python3 msa_to_qwen3_config.py {model_dir} --llama-dir ./llama.cpp")
        return

    print(f"\nConversione GGUF via: {script}")
    print(f"  Input   : {model_dir}")
    print(f"  Output  : {outfile}")
    print(f"  Tipo    : {outtype.upper()}")
    print()

    cmd = [
        sys.executable,
        str(script),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n✗ Conversione fallita (exit {result.returncode})")
        print("  Controlla l'output sopra per i dettagli.")
        print("  Se il problema è 'unknown architecture', verifica che")
        print("  config.json abbia model_type=qwen3 (script già applicato?).")
    else:
        print(f"\n✓ GGUF pronto: {outfile}")
        size_mb = outfile.stat().st_size / 1024 / 1024 if outfile.exists() else 0
        if size_mb:
            print(f"  Dimensione: {size_mb:.1f} MB")

        print(f"""
Prossimi passi:

  Quantizza (opzionale):
    ./llama.cpp/build/bin/llama-quantize {outfile} \\
        msa4b-Q4_K_M.gguf Q4_K_M

  Testa:
    ./llama.cpp/build/bin/llama-cli \\
        -m {outfile} -p "Ciao!" -n 128

  LM Studio:
    Copia {outfile} nella cartella modelli di LM Studio.
""")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patcha MSA-4B config.json → Qwen3 per llama.cpp standard"
    )
    ap.add_argument(
        "model_dir",
        type=Path,
        help="Cartella del modello MSA-4B scaricato da HuggingFace",
    )
    ap.add_argument(
        "--llama-dir",
        type=Path,
        default=None,
        help="Root del repo llama.cpp (per trovare convert_hf_to_gguf.py)",
    )
    ap.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="File GGUF di output (default: <model_dir>/MSA-4B-qwen3-F16.gguf)",
    )
    ap.add_argument(
        "--outtype",
        choices=["f16", "bf16", "f32", "q8_0"],
        default="f16",
        help="Tipo GGUF output (default: f16)",
    )
    ap.add_argument(
        "--no-convert",
        action="store_true",
        help="Patcha solo config.json, non lancia la conversione",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra cosa farebbe senza modificare nulla",
    )
    ap.add_argument(
        "--restore",
        action="store_true",
        help="Ripristina il config.json originale dall'ultimo backup",
    )
    args = ap.parse_args()

    model_dir: Path = args.model_dir.resolve()
    if not model_dir.is_dir():
        sys.exit(f"Directory non trovata: {model_dir}")

    # ── Ripristino backup ────────────────────────────────────────────────────
    if args.restore:
        backups = sorted(model_dir.glob("config.json.msa_backup_*"), reverse=True)
        if not backups:
            sys.exit("Nessun backup trovato in " + str(model_dir))
        latest = backups[0]
        shutil.copy2(latest, model_dir / "config.json")
        print(f"✓ config.json ripristinato da {latest.name}")
        return

    print(f"\n=== MSA-4B → Qwen3 config patch ===\n")

    # ── Patch config.json ────────────────────────────────────────────────────
    patch_config(model_dir, dry_run=args.dry_run)

    if args.dry_run or args.no_convert:
        if not args.dry_run:
            print("--no-convert: config patchato, conversione saltata.")
            print(f"Puoi ora lanciare convert_hf_to_gguf.py manualmente:")
            llama_dir = args.llama_dir or Path("llama.cpp")
            print(
                f"  python3 {llama_dir}/convert_hf_to_gguf.py {model_dir} --outtype f16"
            )
        return

    # ── Conversione GGUF ─────────────────────────────────────────────────────
    outfile = args.outfile or (model_dir / f"MSA-4B-qwen3-{args.outtype.upper()}.gguf")
    run_conversion(model_dir, outfile, args.outtype, args.llama_dir)


if __name__ == "__main__":
    main()
