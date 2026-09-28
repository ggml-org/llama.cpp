import { Compass, Eye, EyeOff, Heart, HeartOff, Trash2, Zap } from '@lucide/svelte';
import { MODEL_DOWNLOAD_ICONS } from '$lib/constants';
import { ModelRowDownloadState } from '$lib/enums';
import { modelsStore, uiStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import { repoOf } from '$lib/utils';

/** Model the configuration pane has open, which a row can be set as the draft of. */
export interface ModelRowDraftTarget {
	id: string;
	label: string;
}

interface RowState {
	/** Backend can load and unload the model. */
	canLoad: boolean;
	/** Download state, when the row stands for a tracked download. */
	download?: ModelRowDownloadState | null;
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
	const { canLoad, download, draftTarget, favorite, isHidden } = state;
	const canBeDraft = canLoad && !!draftTarget && draftTarget.id !== option.id;
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
				onclick: () => handlers.onDelete(option),
				separator: true,
				variant: 'destructive' as const
			},
			viewInDiscover
		];
	}

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
		},
		viewInDiscover
	];
}
