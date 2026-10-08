#include "dispatch-common.h"

#include "dispatch-binary.h"
#include "dispatch-copy.h"
#include "dispatch-flash-attention.h"
#include "dispatch-gated-mul-mat-id.h"
#include "dispatch-gated-mul-mat.h"
#include "dispatch-gather-add.h"
#include "dispatch-get-rows.h"
#include "dispatch-glu.h"
#include "dispatch-mul-mat-id.h"
#include "dispatch-mul-mat.h"
#include "dispatch-rmsnorm.h"
#include "dispatch-rope-set-rows.h"
#include "dispatch-scale.h"
#include "dispatch-unary.h"

namespace ggml::hrx {

void register_common_dispatches(DispatchRegistryBuilder & registry) {
    register_binary_dispatch(registry);
    register_copy_dispatch(registry);
    register_flash_attention_dispatches(registry);
    register_gated_mul_mat_id_dispatches(registry);
    register_gated_mul_mat_dispatches(registry);
    register_gather_add_dispatch(registry);
    register_get_rows_dispatches(registry);
    register_glu_dispatches(registry);
    register_mul_mat_id_dispatches(registry);
    register_mul_mat_dispatches(registry);
    register_rope_set_rows_dispatches(registry);
    register_scale_dispatch(registry);
    register_unary_dispatch(registry);
    register_rmsnorm_dispatches(registry);
}

}  // namespace ggml::hrx
