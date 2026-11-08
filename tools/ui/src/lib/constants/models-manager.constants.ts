/** Labels of the models manager table sections. */

import { ModelsTableGroupKind, ServerModelSource } from '$lib/enums';

export const MODELS_TABLE_GROUP_LABELS: Record<ModelsTableGroupKind, string> = {
	[ModelsTableGroupKind.DOWNLOADING]: 'Downloading',
	[ModelsTableGroupKind.FAVORITES]: 'Favorites',
	[ModelsTableGroupKind.HIDDEN]: 'Hidden models',
	[ModelsTableGroupKind.LOADED]: 'Loaded models',
	[ModelsTableGroupKind.LOCAL]: 'Local models'
};

/**
 * Sticky offset of a group heading under its section header: one section-header
 * row (`py-2` over `text-[13px]` line height) minus the header's bottom border,
 * so the two stacked sticky rows do not show a gap or overlap by a pixel.
 */
export const MODELS_TABLE_GROUP_STICKY_OFFSET = 'top: calc(2.25rem - 1px)';

/**
 * Why a model cannot be deleted from disk, by where the router found it. Only a
 * model from the HF cache is the router's own to remove.
 */
export const MODEL_REMOVE_BLOCKED_REASONS: Record<
	Exclude<ServerModelSource, ServerModelSource.CACHE>,
	string
> = {
	[ServerModelSource.MODELS_DIR]:
		'Found in --models-dir: delete the file from that folder to remove it.',
	[ServerModelSource.PRESET]:
		'Defined in a --models-preset file: remove its section from the preset to delete it.'
};

/** Fallback when the router refuses removal without saying where the model came from. */
export const MODEL_REMOVE_BLOCKED_FALLBACK =
	'The server does not allow deleting this model: it is not from the Hugging Face cache.';
