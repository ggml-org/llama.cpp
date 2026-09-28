/** Labels of the models manager table sections. */

import { ModelsTableGroupKind } from '$lib/enums';

export const MODELS_TABLE_GROUP_LABELS: Record<ModelsTableGroupKind, string> = {
	[ModelsTableGroupKind.DOWNLOADING]: 'Downloading',
	[ModelsTableGroupKind.FAVORITES]: 'Favorites',
	[ModelsTableGroupKind.HIDDEN]: 'Hidden models',
	[ModelsTableGroupKind.LOADED]: 'Loaded models',
	[ModelsTableGroupKind.LOCAL]: 'Local models'
};

/** Panel the models dialog shows. */
export const MODELS_DIALOG_VIEW = {
	DISCOVER: 'discover',
	MANAGE: 'manage',
	PROVIDERS: 'providers'
} as const;

export type ModelsDialogView = (typeof MODELS_DIALOG_VIEW)[keyof typeof MODELS_DIALOG_VIEW];
