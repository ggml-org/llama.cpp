The next feature to implement is support for the `granite-4.0-3b-vision` model (https://huggingface.co/ibm-granite/granite-4.0-3b-vision). This model uses a conditional LoRA adapter. For now, we'll require that the user toggle the adapter on/off manually. You can find a local copy of the model at /Users/ghart/models/ibm-granite/granite-4.0-3b-vision.

The `transformers` implementation for this model is embedded in the model itself, so you can find the full details in `/Users/ghart/models/ibm-granite/granite-4.0-3b-vision/*.py`.

When implementing support for this model here in `llama.cpp`, we need to do the following:

A. Add GGUF conversion support:
    1. Add the necessary architecture/hparam/tensor names and enums in `gguf-py/gguf/`
    2. Add translation support for the model in `convert_hf_to_gguf.py` (both the core text model and the mmproj)
    3. Add support for the adapter conversion in `convert_lora_to_gguf.py`

B. Add `mtmd` support (c++):
    1. Add the corresponding architecture/hparam/tensor names and enums on the c++ side in either `tools/mtmd` or `src/`. I don't _think_ there should be any changes to the enums in the core `src/` library since the text model is already a well supported architecture, but I'm not sure about that.
    2. Ensure support for the vision tokenizer and mmproj model in `tools/mtmd`
    3. Implement support for using the converted LoRA as an adapter

I want to base this work on the current branch which points to the PR for Granite Speech (https://github.com/ggml-org/llama.cpp/pull/22101). I believe these models rely on some of the same QFormer components, so as much as possible those should be reused.

The other main architectural feature for this model is multiple layer injection points from the multimodal projector. You can see this in `modeling.py` with the `deepstack_features` list that gets created during projection, then injected at the configured layers using `masked_scatter`. This injection capability is similar to the Per-Layer-Embeddings (PLE) for Gemma4 or the `n_deepstack_layers` for `qwen3vl`. The main difference is the way the layers are determined for Gemma4, every layer is assumed to have a PLE. For `quen3vl`, it's the first N layers. For Granite 4 vision, the layers are a list specified in config. To implement this, we'll likely need to extend the `llama_hparams` struct's `n_deepstack_layers` to support a layer array (see eg `recurrent_layer_arr` or other `std::array<T, LLAMA_MAX_LAYERS>` arrays). We'll then need to support GGUF values that can be either a list or a single number which would then map back to the single `n_deepstack_layers` value.

I want you to further analyze the model architecture, check with the current base branch for Granite 4 Speech, and come up with a plan to implement support for Granite 4 Vision.