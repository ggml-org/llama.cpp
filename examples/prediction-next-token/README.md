# llama.cpp/examples/prediction-next-token

This directory contains examples demonstrating **next-token prediction** using LLaMA models through [llama.cpp/GGML](https://github.com/ggml-org/llama.cpp).

The tool can be useful for checking and measuring fine tuning results on examples
(Now only on CPU)
---

## Usage

```
prediction-next-token --model <model_path> --prompt <prompt> [--hypothesis <first_word>]
```

or short form:

```
prediction-next-token -m <model_path> -p <prompt> [-h <first_word>]
```

**Example:**

```bash
prediction-next-token -m "models\llama-3.2-1B-q4_k_m-128k.gguf" -p "Who invented E=mc^2?" -h "Einstein"
```

---

### Notes for non-English UTF-8 text (e.g., Russian)

On **Windows**, it is recommended to use **Windows Terminal**:

```
.\prediction-next-token.exe -m "models\llama-3.2-1B-q4_k_m-128k-ru.gguf" -p "Здравствуйте!" -h "Привет"
chcp 65001
```

* This ensures correct handling of UTF-8 characters both for input arguments and output in the console.


---

## Notes on Model Behavior

* The `--hypothesis` argument is optional and specifies expected/necessary the first word to evaluate.
* After fine-tuning on a dataset, the **perplexity** of the model on a test set should decrease over training epochs.


