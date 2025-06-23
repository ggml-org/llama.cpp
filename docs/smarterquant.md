# SmarterQuant Configuration File (`default.smarterquant.json`)

The `default.smarterquant.json` file allows for fine-grained control over the quantization process for specific tensors within a model when using `llama-quantize`. It also provides the necessary information for the inference engine (e.g., `llama-cli`, `llama-server`) to correctly dequantize and use these custom-quantized tensors.

## File Location

When running `llama-quantize` or any inference executable (`llama-cli`, `llama-server`), this JSON file is expected to be present in the **current working directory**.

## Format

The file must be a valid JSON object. Each key in this top-level object is the exact name of a tensor in the model (e.g., `"blk.0.attn_q.weight"`). The value associated with each tensor name is a JSON array containing exactly two elements:

1.  **Compression Types Array (Required for SmarterQuant processing):**
    *   A JSON array of exactly four integers.
    *   These integers correspond to `ggml_type` enum values (e.g., `0` for `GGML_TYPE_F32`, `8` for `GGML_TYPE_Q4_0`, `14` for `GGML_TYPE_Q4_K_M`, etc. Refer to `ggml.h` for the full list of `ggml_type` enums).
    *   The four integers specify the quantization type to be used for the first four 256-column-wide blocks of the tensor, respectively.
        *   `compression_types[0]`: For columns 0-255.
        *   `compression_types[1]`: For columns 256-511.
        *   `compression_types[2]`: For columns 512-767.
        *   `compression_types[3]`: For columns 768-1023.
    *   All subsequent blocks (from column 1024 onwards) will also use the type specified by `compression_types[3]`. This type will also be stored as the main GGUF tensor type.
    *   If this array is empty or not an array of 4 integers, SmarterQuant block-specific quantization will not be applied for this tensor, even if other settings are present.

2.  **Column Permutation Array (Optional):**
    *   A JSON array of integers.
    *   If non-empty, this array defines how the columns of the original tensor should be reordered *before* any quantization (including the block-specific quantization above) is applied.
    *   The length of this array *must* exactly match the number of columns of the tensor (i.e., `tensor->ne[0]`).
    *   The values in the array must be unique integers from `0` to `C-1` (where `C` is the number of columns), representing the original column index.
    *   The new layout will be such that `new_column[j]` takes its data from `original_column[permutation_array[j]]`.
    *   If this array is empty (`[]`), no column permutation is applied.

## Example

```json
{
    "blk.0.attn_q.weight": [
        [8, 9, 12, 13],  // ggml_type for block 0, 1, 2, 3. Block 4+ uses type 13.
                           // (e.g., 8 could be GGML_TYPE_Q4_0, 9 GGML_TYPE_Q4_1, etc.)
        [ /* Large array of column indices, e.g., 0, 2, 1, 5, 4, ... up to tensor_ne0-1 */ ]
    ],
    "blk.1.ffn_down.weight": [
        [14, 14, 14, 14],
        []
    ],
    "output.weight": [
        [2, 2, 2, 2],  // Example: Quantize first four blocks as Q8_0 (assuming 2 maps to Q8_0 in ggml.h)
        []             // No permutation
    ]
}
```

In this example:
-   `blk.0.attn_q.weight`: Will have its columns permuted according to the provided list. Its first 256 columns (after permutation) will be quantized with `ggml_type` 8, the next with type 9, then 12, then 13. Subsequent blocks will also use type 13.
-   `blk.1.ffn_down.weight`: Will not have its columns permuted. All its blocks (first four and subsequent) will be quantized with `ggml_type` 14.
-   `output.weight`: Will not be permuted. All its blocks will be quantized as `ggml_type` 2.

## GGUF Metadata

When `llama-quantize` processes a tensor using instructions from `default.smarterquant.json`, it stores the applied configuration in the GGUF file's metadata for that tensor. This allows the inference engine to correctly dequantize and use the tensor. The following keys are used:

-   `tensor_name.smarterquant.enabled` (boolean): `true` if SmarterQuant processing was applied.
-   `tensor_name.smarterquant.permutation` (string): A JSON string representation of the column permutation array used (e.g., `"[3,0,1,2]"`).
-   `tensor_name.smarterquant.block_types` (string): A JSON string representation of the four compression types used for the initial blocks (e.g., `"[8,9,12,13]"`).

The inference engine will prioritize GGUF metadata. If `default.smarterquant.json` is also present during inference, it's primarily used to get the *original* permutation and block type details if they were not perfectly reconstructible from GGUF metadata alone (though the current implementation aims to store them completely in GGUF).
```
