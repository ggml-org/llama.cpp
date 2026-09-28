import { Eye, EyeOff, Heart, HeartOff, Trash2 } from '@lucide/svelte';
import { modelsStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';

/** Row actions follow the app's dropdown pattern: icon, label, separators, variants. */
export function modelRowActions(
	option: ModelOption,
	favorite: boolean,
	isHidden: boolean,
	onDelete: (option: ModelOption) => void
) {
	return [
		{
			icon: favorite ? HeartOff : Heart,
			label: favorite ? 'Remove from favorites' : 'Add to favorites',
			onclick: () => modelsStore.toggleFavorite(option.model)
		},
		{
			icon: Trash2,
			label: 'Delete from disk',
			onclick: () => onDelete(option),
			separator: true,
			variant: 'destructive' as const
		},
		{
			icon: isHidden ? Eye : EyeOff,
			label: isHidden ? 'Unhide model' : 'Hide model',
			onclick: () => modelsStore.toggleHidden(option.id),
			separator: true
		}
	];
}
