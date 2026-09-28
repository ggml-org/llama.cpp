/** Labels of the models manager table sections. */

import { ModelsTableGroupKind } from '$lib/enums';

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

/** Panel the models dialog shows. */
export const MODELS_DIALOG_VIEW = {
	DISCOVER: 'discover',
	MANAGE: 'manage',
	PROVIDERS: 'providers'
} as const;

export type ModelsDialogView = (typeof MODELS_DIALOG_VIEW)[keyof typeof MODELS_DIALOG_VIEW];
