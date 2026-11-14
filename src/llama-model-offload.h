#pragma once

#include <algorithm>
#include <climits>
#include <stdexcept>

// Internal placement policy. Tensor overrides and backend fallback are applied later.
struct llama_model_offload {
    int n_layer;
    int n_gpu;
    int i_gpu_start;

    static int layer_count(int n_layer_all, int n_layer_nextn, bool load_mtp, int router_layer) {
        if (n_layer_all < 0 || n_layer_all == INT_MAX || n_layer_nextn < 0 || n_layer_nextn > n_layer_all || router_layer < -1 || router_layer >= n_layer_all) {
            throw std::invalid_argument("invalid layer counts for model offload");
        }
        // Router slots and draft-only models use NextN slots without optional MTP loading.
        return load_mtp || router_layer >= 0 || n_layer_nextn == n_layer_all ? n_layer_all : n_layer_all - n_layer_nextn;
    }

    llama_model_offload(int n_layer_all, int n_layer_nextn, bool load_mtp, int n_gpu_layers, bool has_devices, int router_layer = -1) {
        if (n_gpu_layers < 0) {
            throw std::invalid_argument("invalid GPU layer count");
        }
        n_layer = layer_count(n_layer_all, n_layer_nextn, load_mtp, router_layer);
        const int n_requested = std::min(n_gpu_layers, n_layer + 1);
        n_gpu = has_devices ? n_requested : 0;
        i_gpu_start = n_layer + 1 - n_requested;
    }

    int layer_index(int il) const {
        // Unloaded trailing NextN slots use the output placement without shifting the trunk.
        return std::min(il, n_layer);
    }

    bool is_gpu(int il) const {
        return il >= i_gpu_start && il - i_gpu_start < n_gpu;
    }

    int n_repeating_gpu() const {
        return std::max(n_gpu - 1, 0);
    }
};
