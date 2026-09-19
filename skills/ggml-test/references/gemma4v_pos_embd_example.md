# Worked example: `model.position_embeddings` in `tools/mtmd/models/gemma4v.cpp`

This is a complete, verified walkthrough of the workflow in `SKILL.md`, testing the 2-D lookup-table positional embedding block in `clip_graph_gemma4v::build()` (`tools/mtmd/models/gemma4v.cpp`, the `model.position_embeddings` block):

```cpp
{
    const int64_t pos_size = model.position_embeddings->ne[1];
    const size_t  nb1      = ggml_row_size(model.position_embeddings->type, n_embd);

    // positional embeddings are stored as lookup tables (one for x, one for y)
    ggml_tensor * tbl_x = ggml_view_2d(ctx0, model.position_embeddings,
                                         n_embd, pos_size, nb1, 0);
    ggml_tensor * tbl_y = ggml_view_2d(ctx0, model.position_embeddings,
                                         n_embd, pos_size, nb1, pos_size * nb1);

    // ggml_get_rows: [n_embd, n_patches]
    ggml_tensor * emb_x = ggml_get_rows(ctx0, tbl_x, pos_x);
    ggml_tensor * emb_y = ggml_get_rows(ctx0, tbl_y, pos_y);

    inp = ggml_add(ctx0, inp, emb_x);
    inp = ggml_add(ctx0, inp, emb_y);
}
```

## Step 0: read the reference implementation

The HF reference is `Gemma4VisionPatchEmbedder._position_embeddings` in
[`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py):

```python
class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config):
        ...
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(self, pixel_position_ids, padding_positions):
        clamped_positions = pixel_position_ids.clamp(min=0)
        # position_embedding_table: (2, position_embedding_size, hidden_size)
        x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
        position_embeddings = x_emb + y_emb
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings
```

`position_embedding_table` has PyTorch shape `(2, position_embedding_size, hidden_size)` -- axis 0 selects the x- or y-table. Saved as-is with `save_tensors`, gguf-py's axis reversal turns this into a ggml tensor with `ne = [n_embd, pos_size, 2]`: a *3-D*, contiguous tensor where the y-table starts exactly `pos_size * nb[1]` bytes in (that byte offset is `nb[2]` of the 3-D tensor). That's why the C++ code above can treat it as two `ggml_view_2d` slices of a nominally-2-D view -- it's relying on the underlying 3-D tensor being contiguous. This is exactly the kind of stride assumption this skill is good at catching if it were ever wrong.

This test only covers `_position_embeddings` (the lookup-and-add), not the padding mask or the surrounding `input_proj`/conv -- keep each test scoped to one op sequence.

## Step 1: PyTorch reference (`tmp/pt_ref_gemma4v_pos_embd.py`)

```python
import sys
sys.path.insert(0, "skills/ggml-test/scripts")
import numpy as np
import torch
import torch.nn.functional as F
from gguf_io import save_tensors

torch.manual_seed(0)

n_embd    = 8
pos_size  = 6   # config.position_embedding_size
n_patches = 4

# Gemma4VisionPatchEmbedder.position_embedding_table
position_embedding_table = torch.randn(2, pos_size, n_embd)

pos_x = torch.randint(0, pos_size, (n_patches,))
pos_y = torch.randint(0, pos_size, (n_patches,))

# Gemma4VisionPatchEmbedder._position_embeddings (padding path omitted -- not under test)
x_emb = F.embedding(pos_x, position_embedding_table[0])
y_emb = F.embedding(pos_y, position_embedding_table[1])
out = x_emb + y_emb  # (n_patches, n_embd)

save_tensors("tmp/gemma4v_pos_embd_in.gguf", {
    "position_embeddings": position_embedding_table.numpy().astype(np.float32),
    "pos_x": pos_x.numpy().astype(np.int32),
    "pos_y": pos_y.numpy().astype(np.int32),
})
save_tensors("tmp/gemma4v_pos_embd_ref.gguf", {
    "out": out.numpy().astype(np.float32),
})
print("wrote tmp/gemma4v_pos_embd_in.gguf and tmp/gemma4v_pos_embd_ref.gguf")
```

Run: `python3 tmp/pt_ref_gemma4v_pos_embd.py`

## Step 2: ggml test program (`tmp/test_gemma4v_pos_embd.cpp`)

```cpp
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * in_path  = argc > 1 ? argv[1] : "tmp/gemma4v_pos_embd_in.gguf";
    const char * out_path = argc > 2 ? argv[2] : "tmp/gemma4v_pos_embd_out.gguf";

    // 1. load inputs + weights (data already loaded into plain CPU memory)
    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params gguf_params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    struct gguf_context * gguf_ctx = gguf_init_from_file(in_path, gguf_params);
    if (!gguf_ctx) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }

    ggml_tensor * position_embeddings = ggml_get_tensor(ctx_data, "position_embeddings"); // ne = [n_embd, pos_size, 2]
    ggml_tensor * pos_x_data          = ggml_get_tensor(ctx_data, "pos_x");               // ne = [n_patches]
    ggml_tensor * pos_y_data          = ggml_get_tensor(ctx_data, "pos_y");
    if (!position_embeddings || !pos_x_data || !pos_y_data) {
        fprintf(stderr, "missing expected tensor in %s\n", in_path);
        return 1;
    }

    const int64_t n_embd    = position_embeddings->ne[0];
    const int64_t n_patches = pos_x_data->ne[0];

    // 2. build the graph -- copy-pasted verbatim from the `model.position_embeddings`
    //    block in tools/mtmd/models/gemma4v.cpp
    struct ggml_init_params cparams = { /*.mem_size=*/ 16*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(cparams);

    ggml_tensor * pos_x = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_x, "pos_x");
    ggml_set_input(pos_x);

    ggml_tensor * pos_y = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_y, "pos_y");
    ggml_set_input(pos_y);

    ggml_tensor * cur;
    {
        const int64_t pos_size = position_embeddings->ne[1];
        const size_t  nb1      = ggml_row_size(position_embeddings->type, n_embd);

        ggml_tensor * tbl_x = ggml_view_2d(ctx, position_embeddings, n_embd, pos_size, nb1, 0);
        ggml_tensor * tbl_y = ggml_view_2d(ctx, position_embeddings, n_embd, pos_size, nb1, pos_size * nb1);

        ggml_tensor * emb_x = ggml_get_rows(ctx, tbl_x, pos_x);
        ggml_tensor * emb_y = ggml_get_rows(ctx, tbl_y, pos_y);

        cur = ggml_add(ctx, emb_x, emb_y);
    }
    ggml_set_name(cur, "out");
    ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    // 3. allocate + run on CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(galloc, gf);

    ggml_backend_tensor_set(pos_x, pos_x_data->data, 0, ggml_nbytes(pos_x));
    ggml_backend_tensor_set(pos_y, pos_y_data->data, 0, ggml_nbytes(pos_y));

    ggml_backend_graph_compute(backend, gf);

    // 4. dump the output to gguf so a python script can compare it to the pytorch reference
    struct gguf_context * out_ctx = gguf_init_empty();
    struct ggml_init_params out_params = { (size_t) ggml_nbytes(cur) + 1024*1024, nullptr, /*.no_alloc=*/ false };
    struct ggml_context * ctx_out = ggml_init(out_params);
    ggml_tensor * out_t = ggml_dup_tensor(ctx_out, cur);
    ggml_set_name(out_t, "out");
    ggml_backend_tensor_get(cur, out_t->data, 0, ggml_nbytes(cur));
    gguf_add_tensor(out_ctx, out_t);
    gguf_write_to_file(out_ctx, out_path, false);
    printf("wrote %s\n", out_path);

    gguf_free(out_ctx);
    ggml_free(ctx_out);
    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    gguf_free(gguf_ctx);
    ggml_free(ctx_data);
    return 0;
}
```

## Step 3-4: build, run, compare

```bash
skills/ggml-test/scripts/build_and_run.sh tmp/test_gemma4v_pos_embd.cpp
python3 skills/ggml-test/scripts/compare_tensors.py \
    tmp/gemma4v_pos_embd_ref.gguf tmp/gemma4v_pos_embd_out.gguf
```

Verified output:

```
[PASS] out: shape=(4, 8) max_abs=0.000e+00 mean_abs=0.000e+00 rel_l2=0.000e+00
```

An exact match is expected here since every op involved (`ggml_view_2d`, `ggml_get_rows`, `ggml_add`) is exact in f32 with no reduction -- if you see anything above float epsilon on a component like this, suspect a shape/stride bug rather than expected numerical drift.
