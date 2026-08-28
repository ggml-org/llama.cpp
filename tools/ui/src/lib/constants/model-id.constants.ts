/**
 * Parsing of `org/ModelName[-tag][:quant]` style model IDs.
 */

import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';

/** Any sidecar file type: a draft variant or an auxiliary sidecar like mmproj. */
export type ModelSidecar = ModelDraftSidecar | ModelAuxSidecar;

/** Lowercase filename token for each sidecar, e.g. `mtp-Q4_0.gguf`, `mmproj-F16.gguf`. */
export const SIDECAR_FILE_TOKENS: Record<ModelSidecar, string> = {
	[ModelAuxSidecar.MMPROJ]: 'mmproj',
	[ModelDraftSidecar.DFLASH]: 'dflash',
	[ModelDraftSidecar.DSPARK]: 'dspark',
	[ModelDraftSidecar.EAGLE3]: 'eagle3',
	[ModelDraftSidecar.MTP]: 'mtp'
};

const SIDECARS_BY_FILE_TOKEN = Object.fromEntries(
	Object.entries(SIDECAR_FILE_TOKENS).map(([sidecar, token]) => [token, sidecar])
) as Record<string, ModelSidecar>;

/** Map a lowercase filename token (e.g. `mtp`) to its sidecar enum value. */
export function sidecarFromFileToken(token: string): ModelSidecar | null {
	return SIDECARS_BY_FILE_TOKEN[token] ?? null;
}

export function isDraftSidecar(sidecar: ModelSidecar): sidecar is ModelDraftSidecar {
	return (Object.values(ModelDraftSidecar) as string[]).includes(sidecar);
}

export function isAuxSidecar(sidecar: ModelSidecar): sidecar is ModelAuxSidecar {
	return (Object.values(ModelAuxSidecar) as string[]).includes(sidecar);
}

export const MODEL_ID = {
	/**
	 * Matches an activated-parameter-count segment, e.g. `A10B`, `a2.4b`.
	 * The leading `A`/`a` distinguishes it from a regular params segment.
	 */
	ACTIVATED_PARAMS_RE: /^[Aa]\d+(\.\d+)?[BbMmKkTt]$/,

	/** Matches prefix for custom quantization types, e.g. `UD-Q8_K_XL`. */
	CUSTOM_QUANTIZATION_PREFIX_RE: /^UD$/i,
	/** Container format segments to exclude from tags (every model uses these). */
	IGNORED_SEGMENTS: new Set(['GGUF', 'GGML']),
	/** Sentinel value returned by `indexOf` when a substring is not found. */
	NOT_FOUND: -1,
	/** Separates `<org>` from `<model>` in a model ID, e.g. `org/ModelName`. */
	ORG_SEPARATOR: '/',
	/**
	 * Matches a parameter-count segment, e.g. `7B`, `1.5b`, `120M`.
	 * The optional leading `E` covers effective-parameter sizes, e.g. Gemma's
	 * `E2B`/`E4B` (MatFormer models sized by resident params).
	 */
	PARAMS_RE: /^[Ee]?\d+(\.\d+)?[BbMmKkTt]$/,

	/**
	 * Matches a quantization/precision segment, e.g. `Q4_K_M`, `IQ4_XS`, `F16`, `BF16`, `MXFP4`.
	 * Case-insensitive to handle both uppercase and lowercase inputs.
	 */
	QUANTIZATION_SEGMENT_RE: /^(I?Q\d+(_[A-Z0-9]+)*|F\d+|BF\d+|MXFP\d+(_[A-Z0-9]+)*)$/i,

	/** Separates the model path from the quantization tag, e.g. `model:Q4_K_M`. */
	QUANTIZATION_SEPARATOR: ':',

	/** Separates named segments within the model path, e.g. `ModelName-7B-GGUF`. */
	SEGMENT_SEPARATOR: '-',

	/**
	 * Sidecar prefix that wraps a model id with a sidecar type, e.g.
	 * `mtp-<name>.gguf`, `dflash-<name>.gguf`, `dspark-<name>.gguf`,
	 * `eagle3-<name>.gguf`, `mmproj-<name>.gguf`. Captures the bare type
	 * token for typed lookup.
	 */
	SIDECAR_PREFIX_RE: /^(mtp|dflash|dspark|eagle3|mmproj)-(.*)$/i,

	/**
	 * Trailing `-<type>` suffix marking a GGUF with an embedded draft in the
	 * same weight file (MTP) or a sidecar download entry, e.g.
	 * `Hy3-IQ1_M-mtp.gguf`, `Q4_K_M-dspark`. The captured prefix is the
	 * candidate model id; the caller decides whether it looks quantized.
	 */
	SIDECAR_SUFFIX_RE: /^(.*)-(mtp|dflash|dspark|eagle3)$/i,

	/** Matches a trailing weight file extension, e.g. `model.gguf` -> `model`. */
	WEIGHT_EXTENSION_RE: /\.(gguf|ggml)$/i
};
