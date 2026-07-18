/**
 * Model load progress helpers for the /models/sse surfaces
 * (selector row and chat message).
 */

import { MODEL_LOAD_STAGE_LABELS, MODEL_LOAD_TAIL_SHARE } from '$lib/constants';

/**
 * Human label for a model load stage.
 */
export function modelLoadStageLabel(stage: ApiModelLoadStage): string {
	return MODEL_LOAD_STAGE_LABELS[stage];
}

/**
 * Overall load fraction (0.0 -> 1.0) across the declared stage plan.
 * text_model fills [0, 1 - tail], each later phase owns one tail slice.
 */
export function modelLoadFraction(progress: ModelLoadProgress | null): number {
	if (!progress) return 0;

	const { stages, current, value } = progress;
	const tailCount = Math.max(stages.length - 1, 0);
	const textCeiling = 1 - tailCount * MODEL_LOAD_TAIL_SHARE;
	const idx = stages.indexOf(current);

	if (idx <= 0) {
		return value * textCeiling;
	}

	return textCeiling + (idx - 1 + value) * MODEL_LOAD_TAIL_SHARE;
}

/**
 * Single line describing load progress: active stage label and overall percent.
 * Returns null when there is no progress to show.
 */
export function modelLoadProgressText(progress: ModelLoadProgress | null): string | null {
	if (!progress) return null;

	const label = modelLoadStageLabel(progress.current);
	return `${label} ${Math.round(modelLoadFraction(progress) * 100)}%`;
}
