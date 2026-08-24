import type { DraftVariant } from '$lib/constants';
import type { ModelOption } from './models';

/**
 * A draft sidecar model attached to a quant (speculative decoding).
 */
export interface ModelManagerDraft {
	variant: DraftVariant;
	option: ModelOption;
}

/**
 * One quantization of a GGUF repo: main weights plus attached sidecars.
 */
export interface ModelManagerQuant {
	/** Quantization token, e.g. `Q4_K_M`, or null when the entry has none. */
	quant: string | null;
	/** Main model entry (no draft variant). */
	main: ModelOption;
	/** Draft sidecar entries (mtp, dflash, dspark, eagle3). */
	drafts: ModelManagerDraft[];
	/** Multimodal projector sidecar, when registered separately. */
	mmproj: ModelOption | null;
}

/**
 * A GGUF repo by a converter org, e.g. `ggml-org/Qwen3.8-27B-GGUF`.
 */
export interface ModelManagerQuantOrg {
	/** Repo id without the `:quant` tag. */
	repoId: string;
	/** Converter org, e.g. `ggml-org`. */
	orgName: string;
	quants: ModelManagerQuant[];
}

/**
 * The original (non-GGUF) model, e.g. `Qwen/Qwen3.8-27B`.
 */
export interface ModelManagerParent {
	/** Display id: HF base_model when resolved, else the heuristic name. */
	parentId: string;
	quantOrgs: ModelManagerQuantOrg[];
}
