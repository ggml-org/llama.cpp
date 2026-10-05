<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelCapabilities from '../ModelCapabilities.svelte';
	import ModelContext from '../ModelContext.svelte';
	import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
	import ModelId from '../ModelId.svelte';
	import ModelsManagerStatusCell from './ModelsManagerStatusCell.svelte';
	import { modelRowActions } from './row-actions';
	import { configuredContext, downloadProgressFor } from './utils';
	import { MoreHorizontal } from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import { MODEL_ROW_GRID_CLASS, MODEL_ROW_TRAILING_CELL_CLASS } from '$lib/constants';
	import { ModelRowDownloadState } from '$lib/enums';
	import { deviceStore, modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isFavorite: (option: ModelOption) => boolean;
		option: ModelOption;
		onDelete: (option: ModelOption) => void;
		onSelect: (option: ModelOption) => void;
		selected: boolean;
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let { indent = 0, isFavorite, onDelete, onSelect, option, selected }: Props = $props();

	let favorite = $derived(isFavorite(option));
	let isHidden = $derived(modelsStore.isHidden(option.id));
	// live while the download runs, frozen while it is paused
	let downloadProgress = $derived(downloadProgressFor(option.model));
	// a tracked download takes over the status column while it runs
	let download = $derived(
		modelsStore.status.isDownloadPaused(option.model)
			? ModelRowDownloadState.PAUSED
			: modelsStore.status.isDownloadInProgress(option.model)
				? ModelRowDownloadState.DOWNLOADING
				: null
	);
</script>

<div
	class={[
		MODEL_ROW_GRID_CLASS,
		'group relative rounded-md px-2 py-3 transition max-md:px-3 max-md:py-4',
		isHidden && 'opacity-60',
		selected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
	]}
>
	<!-- the row holds a load control and an actions menu, so only the model itself is
	     the button: a button nested in a role="button" row is invalid -->
	<button
		aria-pressed={selected}
		class="flex min-w-0 cursor-pointer items-center gap-3 text-left max-md:gap-2 max-md:pr-9"
		onclick={() => onSelect(option)}
		style="padding-left: {indent}px"
		type="button"
	>
		<ModelAvatar {option} size="size-7 md:size-9" />

		<span class="flex min-w-0 items-center gap-1.25">
			<ModelId
				aliases={option.aliases}
				class="min-w-0 flex-1 max-md:text-sm"
				draftSidecars={option.draftSidecars}
				hideCapabilities
				hideModalities
				modalities={option.modalities}
				modelId={option.model}
				tags={option.tags}
				title={option.model}
			/>

			<!-- a phone has no width for the modality icons, the id needs it more -->
			<ModelCapabilities hideModalities={deviceStore.isMobile} {option} />
		</span>
	</button>

	<ModelContext
		class="justify-self-end max-md:hidden"
		configured={configuredContext(option)}
		{option}
	/>

	<ModelsManagerStatusCell class="max-md:hidden" {download} {option} />

	{#if download}
		<ModelDownloadProgressBar
			downloadedBytes={downloadProgress?.downloadedBytes ?? 0}
			overlay
			totalBytes={downloadProgress?.totalBytes ?? 0}
		/>
	{/if}

	<div class="flex items-center justify-center justify-self-center {MODEL_ROW_TRAILING_CELL_CLASS}">
		<DropdownMenuActions
			actions={modelRowActions(option, favorite, isHidden, onDelete, download)}
			align="end"
			triggerIcon={MoreHorizontal}
			triggerTooltip="Model actions"
		/>
	</div>
</div>
