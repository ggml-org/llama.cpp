import { MODEL_ID, type ModelSidecar, SIDECAR_TOKENS } from '$lib/constants';
import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';

const SIDECAR_TOKEN_SET = new Set<string>(SIDECAR_TOKENS);
const DRAFT_SIDECAR_SET = new Set<string>(Object.values(ModelDraftSidecar));
const AUX_SIDECAR_SET = new Set<string>(Object.values(ModelAuxSidecar));

/** Map a lowercase filename token (e.g. `mtp`) to its sidecar enum value. */
export function sidecarFromFileToken(token: string): ModelSidecar | null {
	return SIDECAR_TOKEN_SET.has(token) ? (token as ModelSidecar) : null;
}

/**
 * Sidecar a download tag points at: the segment after the last dash,
 * e.g. `q4_0-mtp` -> `mtp`, `mmproj` -> `mmproj`. Returns null for quant-only
 * tags and tags whose tail is not a sidecar token.
 */
export function sidecarFromTag(tag: string): ModelSidecar | null {
	const token = tag.toLowerCase().split(MODEL_ID.SEGMENT_SEPARATOR).pop() ?? '';

	return sidecarFromFileToken(token);
}

export function isDraftSidecar(sidecar: ModelSidecar): sidecar is ModelDraftSidecar {
	return DRAFT_SIDECAR_SET.has(sidecar);
}

export function isAuxSidecar(sidecar: ModelSidecar): sidecar is ModelAuxSidecar {
	return AUX_SIDECAR_SET.has(sidecar);
}
