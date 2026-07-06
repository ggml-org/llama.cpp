/**
 * Labels shown while a model loads, keyed by the stage reported on /models/sse.
 */
export const MODEL_LOAD_STAGE_LABELS: Record<ApiModelLoadStage, string> = {
	text_model: 'Loading weights',
	spec_model: 'Loading draft',
	mmproj_model: 'Loading projector'
};

/**
 * Share of the bar reserved for each load phase after text_model.
 * text_model fills the rest, so a plain model reaches 100% on its own.
 */
export const MODEL_LOAD_TAIL_SHARE = 0.1;
