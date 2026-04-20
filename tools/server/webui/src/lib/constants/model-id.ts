/** Sentinel value returned by `indexOf` when a substring is not found. */
export const MODEL_ID_NOT_FOUND = -1;

/** Separates `<org>` from `<model>` in a model ID, e.g. `org/ModelName`. */
export const MODEL_ID_ORG_SEPARATOR = '/';

/** Separates named segments within the model path, e.g. `ModelName-7B-GGUF`. */
export const MODEL_ID_SEGMENT_SEPARATOR = '-';

/** Separates the model path from the quantization tag, e.g. `model:Q4_K_M`. */
export const MODEL_ID_QUANTIZATION_SEPARATOR = ':';

/**
 * Matches a quantization/precision segment, e.g. `Q4_K_M`, `IQ4_XS`, `F16`, `BF16`, `MXFP4`.
 * Case-insensitive to handle both uppercase and lowercase inputs.
 */
export const MODEL_QUANTIZATION_SEGMENT_RE =
	/^(I?Q\d+(_[A-Z0-9]+)*|F\d+|BF\d+|MXFP\d+(_[A-Z0-9]+)*)$/i;

/**
 * Matches prefix for custom quantization types, e.g. `UD-Q8_K_XL`.
 */
export const MODEL_CUSTOM_QUANTIZATION_PREFIX_RE = /^UD$/i;

/**
 * Matches a parameter-count segment, e.g. `7B`, `1.5b`, `120M`.
 */
export const MODEL_PARAMS_RE = /^\d+(\.\d+)?[BbMmKkTt]$/;

/**
 * Matches an activated-parameter-count segment, e.g. `A10B`, `a2.4b`.
 * The leading `A`/`a` distinguishes it from a regular params segment.
 */
export const MODEL_ACTIVATED_PARAMS_RE = /^[Aa]\d+(\.\d+)?[BbMmKkTt]$/;

/**
 * Container format segments to exclude from tags (every model uses these).
 */
export const MODEL_IGNORED_SEGMENTS = new Set(['GGUF', 'GGML']);
