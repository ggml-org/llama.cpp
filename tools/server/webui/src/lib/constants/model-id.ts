/** Sentinel value returned by `indexOf` when a substring is not found. */
export const MODEL_ID_NOT_FOUND = -1;

/** Separates `<org>` from `<model>` in a model ID, e.g. `org/ModelName`. */
export const MODEL_ID_ORG_SEPARATOR = '/';

/** Separates named segments within the model path, e.g. `ModelName-7B-GGUF`. */
export const MODEL_ID_SEGMENT_SEPARATOR = '-';

/** Separates the model path from the quantization tag, e.g. `model:Q4_K_M`. */
export const MODEL_ID_QUANTIZATION_SEPARATOR = ':';

/**
 * Matches a trailing ALL-CAPS format segment, e.g. `GGUF`, `BF16`, `Q4_K_M`.
 * Must be at least 2 uppercase letters, optionally followed by uppercase letters or digits.
 */
export const MODEL_FORMAT_SEGMENT_RE = /^[A-Z]{2,}[A-Z0-9]*$/;

/**
 * Matches a parameter-count segment, e.g. `7B`, `1.5b`, `120M`.
 */
export const MODEL_PARAMS_RE = /^\d+(\.\d+)?[BbMmKkTt]$/;

/**
 * Matches an activated-parameter-count segment, e.g. `A10B`, `A2.4b`.
 * The leading `A` distinguishes it from a regular params segment.
 */
export const MODEL_ACTIVATED_PARAMS_RE = /^A\d+(\.\d+)?[BbMmKkTt]$/;
