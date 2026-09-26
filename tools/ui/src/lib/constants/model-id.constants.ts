/**
 * Parsing of `org/ModelName[-tag][:quant]` style model IDs.
 */

import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';

/** Any sidecar file type: a draft variant or an auxiliary sidecar like mmproj. */
export type ModelSidecar = ModelDraftSidecar | ModelAuxSidecar;

/** All sidecar filename tokens: the bare lowercase enum values, e.g. `mtp`, `mmproj`. */
export const SIDECAR_TOKENS: string[] = [
	...Object.values(ModelDraftSidecar),
	...Object.values(ModelAuxSidecar)
];

/** Separator between token alternatives in the sidecar regexes. */
const REGEX_ALTERNATION_SEPARATOR = '|';
const SIDECAR_TOKEN_ALTERNATION = SIDECAR_TOKENS.join(REGEX_ALTERNATION_SEPARATOR);

export const MODEL_ID = {
	/**
	 * Matches an activated-parameter-count segment, e.g. `A10B`, `a2.4b`.
	 * The leading `A`/`a` distinguishes it from a regular params segment.
	 */
	ACTIVATED_PARAMS_REGEX: /^[Aa]\d+(\.\d+)?[BbMmKkTt]$/,

	/** Matches prefix for custom quantization types, e.g. `UD-Q8_K_XL`. */
	CUSTOM_QUANTIZATION_PREFIX_REGEX: /^UD$/i,
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
	PARAMS_REGEX: /^[Ee]?\d+(\.\d+)?[BbMmKkTt]$/,

	/**
	 * Matches a quantization/precision segment, e.g. `Q4_K_M`, `IQ4_XS`, `F16`, `BF16`, `MXFP4`.
	 * Case-insensitive to handle both uppercase and lowercase inputs.
	 */
	QUANTIZATION_SEGMENT_REGEX: /^(I?Q\d+(_[A-Z0-9]+)*|F\d+|BF\d+|MXFP\d+(_[A-Z0-9]+)*)$/i,

	/** Separates the model path from the quantization tag, e.g. `model:Q4_K_M`. */
	QUANTIZATION_SEPARATOR: ':',

	/** Separates named segments within the model path, e.g. `ModelName-7B-GGUF`. */
	SEGMENT_SEPARATOR: '-',

	/**
	 * Sidecar token between name segments, e.g. `Model-mtp-Q4_0.gguf`,
	 * `model-eagle3-BF16.gguf`. Captures the name head and tail around the
	 * token; same case-insensitive rule as the prefix form.
	 */
	SIDECAR_INFIX_REGEX: new RegExp(`^(.*)-(${SIDECAR_TOKEN_ALTERNATION})-(.+)$`, 'i'),

	/**
	 * Sidecar prefix that wraps a model id with a sidecar type, e.g.
	 * `mtp-<name>.gguf`, `dflash-<name>.gguf`, `dspark-<name>.gguf`,
	 * `eagle3-<name>.gguf`, `mmproj-<name>.gguf`. Captures the bare type
	 * token for typed lookup.
	 *
	 * The token matches case-insensitively (real repos ship uppercase
	 * heads, e.g. `Model-MTP-BF16.gguf`) and is normalized through
	 * `sidecarFromFileToken`; the server's filename grammar
	 * (common/download.cpp) matches the same segments.
	 */
	SIDECAR_PREFIX_REGEX: new RegExp(`^(${SIDECAR_TOKEN_ALTERNATION})-(.*)$`, 'i'),

	/**
	 * Trailing `-<type>` suffix marking a GGUF with an embedded draft in the
	 * same weight file (MTP) or a sidecar download entry, e.g.
	 * `Hy3-IQ1_M-mtp.gguf`, `Q4_K_M-dspark`. An optional `-draft` tail covers
	 * standalone sidecar files, e.g. `Model-mtp-draft.gguf`. The captured
	 * prefix is the candidate model id; the caller decides whether it looks
	 * quantized. Case-insensitive, like the prefix form.
	 */
	SIDECAR_SUFFIX_REGEX: new RegExp(`^(.*)-(${SIDECAR_TOKEN_ALTERNATION})(-draft)?$`, 'i'),

	/** Matches a trailing weight file extension, e.g. `model.gguf` -> `model`. */
	WEIGHT_EXTENSION_REGEX: /\.(gguf|ggml)$/i
};
