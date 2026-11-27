// Tables the way models write them: mixed column alignment, code and links in
// cells, numbers meant to line up, a cell that wraps, escaped pipes and empty
// cells, and one table too wide for the chat column.
const tick = '`';

export const MARKDOWN_TABLES_MD = `Here are a few tables with different alignments and content:

## Server flags

| Flag | Default | Description |
| :--- | :---: | --- |
| ${tick}--ctx-size${tick} | 4096 | Context size in tokens |
| ${tick}--parallel${tick} | 1 | Slots that serve requests at once |
| ${tick}--jinja${tick} | on | Use the model's own chat template |

## Quantization types

| Type | Bits per weight | Size (7B) | Notes |
| :--- | ---: | ---: | :--- |
| ${tick}Q4_K_M${tick} | 4.85 | 4.07 GB | A good default for most models |
| ${tick}Q5_K_M${tick} | 5.69 | 4.78 GB | A little more quality for a little more memory |
| ${tick}Q8_0${tick} | 8.50 | 7.16 GB | Close to the original weights, see [the docs](https://github.com/ggml-org/llama.cpp) |
| ${tick}F16${tick} | 16.00 | 13.50 GB | **Full** half precision |

## Build status

| Package | Status | Version | Notes |
| :--- | :--- | ---: | :--- |
| ${tick}@ggml-org/llama-ui${tick} | ✅ Passing | 0.4.2 | See [ui/lib/ui](https://github.com/ggml-org/llama.cpp/tree/master/ui/lib/ui) |
| ${tick}llama-storybook${tick} | ⚠️ Flaky | 0.3.0 | **Needs** a rerun on CI |
| ${tick}llama-brand${tick} | ❌ Failing | 0.1.9 | Missing ${tick}BRAND.md${tick} |

## Cells that hold more than a word

| Input | Output | Escaped pipe |
| :--- | :--- | :--- |
| ${tick}a \\| b${tick} | ${tick}a or b${tick} | ✔ |
| | (empty input) | ✘ |
| A list | <ul><li>first</li><li>second</li></ul> | — |
| Two lines | first<br>second | — |

## A wide table

| Model | Family | Parameters | Context | Quantization | Prompt t/s | Generation t/s | Memory | Backend | Notes |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| gpt-oss-20b | GPT-OSS | 20.9B | 131072 | MXFP4 | 1890.4 | 74.2 | 12.1 GB | Metal | Reasoning effort low, medium and high |
| Qwen3-8B | Qwen3 | 8.2B | 40960 | Q4_K_M | 1312.7 | 58.9 | 5.0 GB | Metal | Thinking can be turned off per request |
`;
