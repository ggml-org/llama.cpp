# HYV3Speech: simultaneous speech translation

HYV3Speech (Hugging Face architecture `HYV3SpeechForCausalLM`, GGUF arch `hy_v3`) is a
simultaneous interpretation model. It listens to a stream of audio and translates what it
has heard so far, instead of waiting for the speaker to finish.

> [!IMPORTANT]
>
> Two things about this model are unusual, and both change the output if you get them wrong:
>
> - **It is fed turn by turn.** The model was trained on 1600 ms of audio followed by
>   800 ms per turn, and it answers each turn before hearing the next one. `llama-mtmd-cli`
>   detects this from the projector and does it automatically. `llama-server` cannot run
>   these GGUFs yet, because it still expects the projector in a separate `--mmproj`
>   file; use `llama-mtmd-cli` until turn-by-turn serving lands.
> - **The end-of-turn token has to be known.** A GGUF built for this model carries
>   `tokenizer.ggml.eot_token_id = 120008`, which puts the token into the EOG set. If
>   yours does not, add `--override-kv tokenizer.ggml.eot_token_id=int:120008`, otherwise
>   the model repeats the same turn until it runs out of context.

## Quick start

The published GGUFs are **single file**: each one carries the text model together with its
own audio tower, so `--mmproj` is not needed. The two checkpoints do not share a tower -
every GGUF carries the one belonging to its own checkpoint.

- Translation only (`tgt_only`): [https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF) - `Hy-SimulMT-30B-A3B-Q8_0.gguf`, `Hy-SimulMT-30B-A3B-Q4_K_M.gguf`
- Source text and translation (`src_tgt`): [https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF) - `Hy-SimulMT-30B-A3B-Interleave-Q8_0.gguf`, `Hy-SimulMT-30B-A3B-Interleave-Q4_K_M.gguf`

Model card, inference service and training tools: <https://github.com/Tencent-Hunyuan/Hy-SimulMT>

```bash
cmake -B build
cmake --build build --config Release --target llama-mtmd-cli

# from a local file
./build/bin/llama-mtmd-cli \
  -m /path/to/Hy-SimulMT-30B-A3B-Interleave-Q8_0.gguf \
  --audio your.wav \
  -c 16384 -n 2048 --temp 0 \
  -p '<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。<__media__><｜hy_Assistant｜><think></think><｜hy_place▁holder▁no▁790｜>'

# or straight from the hub
./build/bin/llama-mtmd-cli \
  -hf tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF:Hy-SimulMT-30B-A3B-Interleave-Q8_0.gguf \
  --audio your.wav \
  -c 16384 -n 2048 --temp 0 \
  -p '<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。<__media__><｜hy_Assistant｜><think></think><｜hy_place▁holder▁no▁790｜>'
```

Audio is resampled to 16 kHz mono before the mel is computed, so any input sample rate and
channel count loads. For long inputs, prefer 16 kHz mono: a long 48 kHz stereo file aborts
once the audio has been consumed (see [Known issues](#known-issues)).

## Prompt

`-p` must contain the media marker `<__media__>`. Everything before it is the user prompt
and is fed once; everything after it is the assistant header and is replayed at the head of
every turn:

```
<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。<__media__><｜hy_Assistant｜><think></think><｜hy_place▁holder▁no▁790｜>
```

The checkpoint comes in two flavours and they differ only in the tail of the prompt:

| Checkpoint | Published as | Tail after `<__media__>` | Answer per turn |
| --- | --- | --- | --- |
| `src_tgt` | `Hy-SimulMT-30B-A3B-Interleave-GGUF` | `<think></think>` + `<｜hy_place▁holder▁no▁790｜>` | source text, `<｜hy_place▁holder▁no▁791｜>`, target text |
| `tgt_only` | `Hy-SimulMT-30B-A3B-GGUF` | `<think></think>` | target text only |

`<｜hy_place▁holder▁no▁790｜>` (id 120808) is the source-branch marker. The `src_tgt` checkpoint was trained
with it at the head of every assistant turn, so it has to be seeded there at inference
time; putting it in the prompt is how you do that, and it is why the two checkpoints need
different prompts. If you leave it out, the source branch degenerates into a copy of the
target, or disappears entirely. Do **not** add it for `tgt_only`: that checkpoint was
trained without any branch marker and returns empty turns if one is injected.

The instruction text has to match the direction of the data you are running. Half of the
samples this model was trained on ask for translation *into English*; writing a Chinese
instruction for those sends the model off in the wrong direction.

## Output

One line per turn, printed as the turns are answered. Inside a line, `<｜hy_place▁holder▁no▁791｜>`
(id 120809) separates the source branch from the target branch:

```
didn't do that<｜hy_place▁holder▁no▁791｜>他没有那样做，
```

A turn with nothing before the separator (or an entirely empty line) is a WAIT: the model
heard the audio but is not ready to commit to a translation yet. That is expected and
happens on most turns.

## Building the projector yourself

The published GGUFs already come merged. The script below rebuilds the projector on its
own, which is what you want if you are re-converting from a checkpoint or want the
two-file (`-m` + `--mmproj`) layout instead.


The projector is not produced by `convert_hf_to_gguf.py`; use the standalone script, which
reads the `audio_tower.*` tensors straight out of the merged checkpoint:

```bash
python tools/mtmd/legacy-models/convert_hyv3_speech_mmproj.py <hf_dir> mmproj-hyv3-speech-f32.gguf
python convert_hf_to_gguf.py <hf_dir> --outfile hymt-A3b-src_tgt-f16.gguf --outtype f16
./build/bin/llama-quantize hymt-A3b-src_tgt-f16.gguf hymt-A3b-src_tgt-Q8_0.gguf Q8_0
```

Every projector tensor is stored as F32. The source weights are BF16 and downcasting them
to F16 would drop mantissa bits for no saving worth having.

## Known issues

- A long **48 kHz stereo** file aborts in the tokenizer once the audio has been consumed.
  Everything the model translated is already on stdout by then, so it is cosmetic, but
  convert to 16 kHz mono if it bothers you.
- `-n` is a **global** token budget for the whole file, not a per-turn one.
- `llama-server` cannot run these GGUFs yet: it expects the projector in a separate
  `--mmproj` file, so a single-file GGUF loads as a plain text model and its audio
  input is silently unavailable. Serving support, including turn-by-turn feeding, is a
  follow-up. Until then use `llama-mtmd-cli`.
- An older `llama.cpp` build cannot load these files at all: it does not know that the
  leftover tensors belong to the packaged projector and fails with `wrong number of
  tensors`. Use a build that includes this support.
