import { Compass, Eye, EyeOff, Heart, HeartOff, Trash2 } from '@lucide/svelte';
import { MODEL_DOWNLOAD_ICONS } from '$lib/constants';
import { ModelRowDownloadState } from '$lib/enums';
import { modelsStore, uiStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import { repoOf } from '$lib/utils';

/** Row actions follow the app's dropdown pattern: icon, label, separators, variants. */
export function modelRowActions(
	option: ModelOption,
	favorite: boolean,
	isHidden: boolean,
	onDelete: (option: ModelOption) => void,
	/** Download state, when the row stands for a tracked download. */
	download?: ModelRowDownloadState | null
) {
	const viewInDiscover = {
		icon: Compass,
		label: 'View in Discover',
		// the details pane is keyed by repo, not by `<repo>:<quant>`
		onclick: () => uiStore.openModelsDiscover(repoOf(option.model)),
		separator: true
	};

	// a tracked download is paused, resumed or dropped: it is not loaded, hidden or
	// drafted. Deleting it stops the download and removes what is on disk, which is
	// what the same wording offers on a cached model
	if (download) {
		const isPaused = download === ModelRowDownloadState.PAUSED;

		return [
			{
				icon: isPaused ? MODEL_DOWNLOAD_ICONS.resume : MODEL_DOWNLOAD_ICONS.pause,
				label: isPaused ? 'Resume downloading' : 'Pause downloading',
				onclick: () =>
					void (isPaused
						? modelsStore.status.downloadModel(option.model)
						: modelsStore.status.pauseDownload(option.model))
			},
			{
				icon: Trash2,
				label: 'Delete from disk',
				onclick: () => onDelete(option),
				separator: true,
				variant: 'destructive' as const
			},
			viewInDiscover
		];
	}

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
		},
		viewInDiscover
	];
}
