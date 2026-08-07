# Qwen3.6-35B-A3B automatic S8 recipes

The recipe remains script-only and uses existing GGUF tensor types. It can either preserve an existing historical stock Q4_0 tensor map or derive a new base map directly from the BF16 source.

`scripts/build-qwen35-q4-0-s8.sh` performs:

1. raw HF directory -> BF16 main GGUF, retaining embedded MTP;
2. raw HF directory -> direct Q8_0 vision projector;
3. a live `llama-quantize --dry-run ... Q4_K_M` sensitivity pass;
4. a base Q4_0 tensor-map plan, with model-specific Q4_0 -> Q8_0 promotions;
5. either the quality-isolation `fixed`, V620-native `native`, or BF16-error-ranked `auto` stage;
6. optional imatrix-weighted quantization.

## Stages

### `fixed` (default)

With `--stock-q4-0`, this preserves the historical stock precision recipe and adds only its Q4_K_M-selected promotions (29 for the current Qwen3.6 map). Without that option, it uses the current BF16 Q4_0 dry-run as the base and derives the promotion count afresh. It retains base Q5_0, Q4_1, and Q6_K entries. BF16 entries are emitted as F32 by llama-quantize, which is a precision increase.

### `auto`

This derives the base map from BF16, estimates weighted Q4_0 versus Q8_0 reconstruction error for every eligible tensor, and promotes the highest-value tensors until the configured Q8 byte fraction is reached. Use `--auto-q8-fraction PCT` to control the size/quality tradeoff, and `--auto-max-tensor-mib N` to restrict upgrades to low-bandwidth tensors. SSM alpha/beta tensors are preserved as F32.

### `native`

This starts from the same derived or supplied base map and maps the remaining Q5_0, Q4_1, and Q6_K tensors to Q8_0. BF16 entries become F32. The final type set is F32, Q4_0, and Q8_0; counts are reported by the generated plan and can differ between models.

The native stage may require `--allow-large-q8`; inspect the generated summary first.

## Usage

Build `llama-quantize` first. By default the base map is derived from the BF16 source:

```bash
cd /home/edwin/llama.cpp-rdna2
scripts/build-qwen35-q4-0-s8.sh \
  --input /home/edwin/models/Qwen3.6-35B-A3B-raw \
  --out-dir /home/edwin/models/qwen35-q4-0-s8-fixed \
  --stage fixed \
  --threads 24 \
  --imatrix /home/edwin/models/qwen35-imatrix/imatrix_unsloth.gguf_file
```

Use `--stage native` for the V620-native build after validating the fixed stage. Use `--stage auto --auto-q8-fraction 25` for the BF16-ranked policy. Use `--stock-q4-0 PATH` only to reproduce a historical stock map. Use `--plan-only` to inspect the derived map before quantizing.

## Automatic candidate sweep

`scripts/optimize-qwen35-s8.sh` builds and evaluates selected candidates (`stock`, `fixed`, `native`, or `auto`) using the same BF16 source. It measures code/Wikitext KLD, PP4096, TG512, and direct `llama-server` MTP TG512, then recommends the best candidate subject to a KLD floor. Use `--stages auto` to evaluate the BF16-ranked policy.

```bash
scripts/optimize-qwen35-s8.sh \
  --input /home/edwin/models/Qwen3.6-35B-A3B-raw \
  --bf16 /home/edwin/models/qwen35-q4-0-s8/Qwen3.6-35B-A3B-MTP-BF16.gguf \
  --out-root /home/edwin/models/qwen35-sweep \
  --code-kld-base /home/edwin/models/qwen35-kld-code20.kld \
  --wiki-kld-base /home/edwin/models/qwen35-kld-wiki20.kld \
  --objective balanced \
  --threads 24
```

Use `--objective mtp`, `decode`, or `prompt` to change the speed priority. `--quality-tolerance PCT` permits a controlled KLD increase over the stock candidate. By default the sweep derives its base map from the BF16 Q4_0 dry-run; pass `--stock-q4-0 PATH` only when reproducing a historical stock map.

The imatrix is auto-detected at `$HOME/models/qwen35-imatrix/imatrix_unsloth.gguf_file`; use `--no-imatrix` to disable it. It changes quantization values but not the Q4_K_M selection; the auto ranker additionally uses its per-column importance weights.

The script refuses already-quantized input, retains BF16 by default, and validates the final stage-specific tensor set. Use `--remove-bf16` only after validation.