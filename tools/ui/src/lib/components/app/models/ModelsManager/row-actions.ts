import { Eye, EyeOff, Heart, HeartOff, Trash2, Zap } from '@lucide/svelte';
import { modelsStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';

/** Model the configuration pane has open, which a row can be set as the draft of. */
export interface ModelRowDraftTarget {
	id: string;
	label: string;
}

interface RowState {
	/** Backend can load and unload the model. */
	canLoad: boolean;
	/** Model the pane has open, when one is selected. */
	draftTarget?: ModelRowDraftTarget | null;
	favorite: boolean;
	isHidden: boolean;
}

interface RowHandlers {
	onDelete: (option: ModelOption) => void;
	onUseAsDraft?: (draft: ModelOption, targetId: string) => void;
}

/** Row actions follow the app's dropdown pattern: icon, label, separators, variants. */
export function modelRowActions(option: ModelOption, state: RowState, handlers: RowHandlers) {
	const { canLoad, draftTarget, favorite, isHidden } = state;
	const canBeDraft = canLoad && !!draftTarget && draftTarget.id !== option.id;

	return [
		...(canBeDraft
			? [
					{
						icon: Zap,
						label: `Use as draft for ${draftTarget.label}`,
						onclick: () => handlers.onUseAsDraft?.(option, draftTarget.id),
						separator: true
					}
				]
			: []),
		{
			icon: favorite ? HeartOff : Heart,
			label: favorite ? 'Remove from favorites' : 'Add to favorites',
			onclick: () => modelsStore.toggleFavorite(option.model)
		},
		...(canLoad
			? [
					{
						icon: Trash2,
						label: 'Delete from disk',
						onclick: () => handlers.onDelete(option),
						separator: true,
						variant: 'destructive' as const
					}
				]
			: []),
		{
			icon: isHidden ? Eye : EyeOff,
			label: isHidden ? 'Unhide model' : 'Hide model',
			onclick: () => modelsStore.toggleHidden(option.id),
			separator: true
		}
	];
}
