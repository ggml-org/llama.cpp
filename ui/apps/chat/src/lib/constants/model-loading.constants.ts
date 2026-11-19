/**
 * Labels shown while a model loads, keyed by the stage reported on /models/sse.
 */
export const MODEL_LOAD_STAGE_LABELS: Record<ApiModelLoadStage, string> = {
	mmproj_model: 'Loading projector',
	spec_model: 'Loading draft',
	text_model: 'Loading weights'
};

/**
 * Share of the bar reserved for each load phase after text_model.
 * text_model fills the rest, so a plain model reaches 100% on its own.
 */
export const MODEL_LOAD_TAIL_SHARE = 0.1;

/** Load parameters a model falls back to when the server reports nothing. */
export const LOAD_DEFAULTS = {
	batchSize: 2048,
	contextLength: 8192,
	cpuThreads: 13,
	gpuOffload: 42,
	speculativeDecoding: 'off',
	ubatchSize: 512
};

/** Sampling parameters a model falls back to. */
export const SAMPLING_DEFAULTS = {
	minP: 0.05,
	repeatPenalty: 1.1,
	temperature: 1,
	topK: 64,
	topP: 0.95
};

/** Speculative decoding modes the load form offers. */
export const SPECULATIVE_OPTIONS = ['off', 'draft-model'];
