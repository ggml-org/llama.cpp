/**
 * Parsing of `org/ModelName[-tag][:quant]` style model IDs.
 */

export const MODEL_ID = {
	/**
	 * Matches an activated-parameter-count segment, e.g. `A10B`, `a2.4b`.
	 * The leading `A`/`a` distinguishes it from a regular params segment.
	 */
	ACTIVATED_PARAMS_RE: /^[Aa]\d+(\.\d+)?[BbMmKkTt]$/,

	/** Matches prefix for custom quantization types, e.g. `UD-Q8_K_XL`. */
	CUSTOM_QUANTIZATION_PREFIX_RE: /^UD$/i,
	/**
	 * Sidecar prefix that wraps a model id with a draft/aux variant, e.g.
	 * `mtp-<name>.gguf`, `dflash-<name>.gguf`, `dspark-<name>.gguf`,
	 * `eagle3-<name>.gguf`, `mmproj-<name>.gguf`. Captures the bare variant
	 * token for typed lookup.
	 */
	DRAFT_VARIANT_PREFIX_RE: /^(mtp|dflash|dspark|eagle3|mmproj)-(.*)$/i,
	/**
	 * Trailing `-<variant>` suffix marking a GGUF with an embedded draft in the
	 * same weight file (MTP) or a sidecar download entry, e.g.
	 * `Hy3-IQ1_M-mtp.gguf`, `Q4_K_M-dspark`. The captured prefix is the
	 * candidate model id; the caller decides whether it looks quantized.
	 */
	DRAFT_VARIANT_SUFFIX_RE: /^(.*)-(mtp|dflash|dspark|eagle3)$/i,

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
	 * Matches a quantization/precision segment, e.g. `Q4_K_M`, `IQ4_XS`, `TQ1_0`,
	 * `F16`, `BF16`, `MXFP4`. Case-insensitive to handle both cases.
	 */
	QUANTIZATION_SEGMENT_RE:
		/^(I?Q\d+(_[A-Z0-9]+)*|TQ\d+(_[A-Z0-9]+)*|F\d+|BF\d+|MXFP\d+(_[A-Z0-9]+)*)$/i,

	/** Separates the model path from the quantization tag, e.g. `model:Q4_K_M`. */
	QUANTIZATION_SEPARATOR: ':',

	/** Separates named segments within the model path, e.g. `ModelName-7B-GGUF`. */
	SEGMENT_SEPARATOR: '-',

	/** Matches a trailing weight file extension, e.g. `model.gguf` -> `model`. */
	WEIGHT_EXTENSION_RE: /\.(gguf|ggml)$/i
};

/**
 * Auxiliary / draft variant segments that show up in GGUF filenames and HF repo IDs.
 * - `mtp`     multi-token-prediction draft model
 * - `dflash`  diffusion-flash draft
 * - `dspark`  DSpark speculative-decoding draft
 * - `eagle3`  Eagle3 speculative-decoding draft
 * - `mmproj`  multimodal projector sidecar
 */
export type DraftVariant = 'mtp' | 'dflash' | 'dspark' | 'eagle3' | 'mmproj';
