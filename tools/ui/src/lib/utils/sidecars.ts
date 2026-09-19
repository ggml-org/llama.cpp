import { type ModelSidecar, SIDECAR_TOKENS } from '$lib/constants';
import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';

const SIDECAR_TOKEN_SET = new Set<string>(SIDECAR_TOKENS);
const DRAFT_SIDECAR_SET = new Set<string>(Object.values(ModelDraftSidecar));
const AUX_SIDECAR_SET = new Set<string>(Object.values(ModelAuxSidecar));

/** Map a lowercase filename token (e.g. `mtp`) to its sidecar enum value. */
export function sidecarFromFileToken(token: string): ModelSidecar | null {
	return SIDECAR_TOKEN_SET.has(token) ? (token as ModelSidecar) : null;
}

export function isDraftSidecar(sidecar: ModelSidecar): sidecar is ModelDraftSidecar {
	return DRAFT_SIDECAR_SET.has(sidecar);
}

export function isAuxSidecar(sidecar: ModelSidecar): sidecar is ModelAuxSidecar {
	return AUX_SIDECAR_SET.has(sidecar);
}
