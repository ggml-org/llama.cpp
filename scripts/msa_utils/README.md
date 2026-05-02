# MSA-4B llama.cpp patch

Supporto per **EverMind-AI/MSA-4B** (Memory Sparse Attention, arXiv:2603.23516)
in llama.cpp. Funziona in **degraded mode**: il backbone Qwen3-4B gira completo,
i router MSA (layer 18-35) sono caricati ma non collegati al graph —
il memory bank da 100M token non è attivo.

## Contenuto

```
src/
  llama-arch.h        ← LLM_ARCH_MSA aggiunto all'enum
  llama-arch.cpp      ← "msa" nella mappa nomi + rope_type NEOX
  llama-model.cpp     ← hparams loader + tensor loader + build switch
  llama-quant.cpp     ← quantizzazione MSA supportata
  models/
    models.h          ← dichiarazione llm_build_msa
    msa.cpp           ← implementazione graph builder (copia Qwen3)
convert_msa_to_gguf.py   ← converter HF safetensors → GGUF
apply_msa_patch.py       ← patcher automatico (per versioni future)
```

## Installazione

### 1. Copia i file in llama.cpp

```bash
LLAMA=/path/to/llama.cpp

cp src/llama-arch.h       $LLAMA/src/
cp src/llama-arch.cpp     $LLAMA/src/
cp src/llama-model.cpp    $LLAMA/src/
cp src/llama-quant.cpp    $LLAMA/src/
cp src/models/models.h    $LLAMA/src/models/
cp src/models/msa.cpp     $LLAMA/src/models/
```

### 2. Ricompila

```bash
cd $LLAMA
cmake --build build -j$(nproc)
```

### 3. Scarica il modello da HuggingFace

```bash
pip install huggingface_hub
huggingface-cli download EverMind-AI/MSA-4B --local-dir models/MSA-4B
```

### 4. Converti in GGUF

```bash
pip install gguf safetensors torch

python3 convert_msa_to_gguf.py models/MSA-4B \
    --outfile models/MSA-4B/MSA-4B-F16.gguf \
    --outtype f16
```

### 5. Testa

```bash
$LLAMA/build/bin/llama-cli \
    -m models/MSA-4B/MSA-4B-F16.gguf \
    -ngl 99 --temp 0.6 --top-p 0.9 -n 256 \
    -p "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of Italy?<|im_end|>
<|im_start|>assistant
"
```

### Quantizzazione (opzionale)

```bash
$LLAMA/build/bin/llama-quantize \
    models/MSA-4B/MSA-4B-F16.gguf \
    models/MSA-4B/MSA-4B-Q4_K_M.gguf \
    Q4_K_M
```

## Note tecniche

- **Norm weights in F32**: i pesi 1D (norme) vengono salvati in F32 anche con
  `--outtype f16` perché `ggml_rms_norm` produce output F32 e `ggml_mul` non
  supporta tipi misti F32×F16.
- **Router tensors esclusi**: i 36 tensori `router_{q,k}_proj.0.weight` non
  vengono scritti nel GGUF — il tensor loader non li registra e causerebbero
  "wrong number of tensors".
- **RoPE type**: `LLAMA_ROPE_TYPE_NEOX` identico a Qwen3.
- **QK norm**: reshape 2D→norm→3D per evitare il crash CUDA broadcast su
  tensori non-contigui.
- **Chat template**: usa il formato Qwen3 con `<|im_start|>` / `<|im_end|>`.
  Il `--chat-template qwen3` in modalità `-cnv` non inserisce il prefisso
  `<think>` necessario — usa `-p` con prompt esplicito.

## Requisiti

- llama.cpp build b9007 o successivo (architettura class-based)
- CUDA opzionale (testato su RTX 4090)
- Python 3.10+: `pip install gguf safetensors torch`
