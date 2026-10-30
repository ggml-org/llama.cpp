<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelCapabilities from '../ModelCapabilities.svelte';
	import ModelContext from '../ModelContext.svelte';
	import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
	import ModelDraftSubline from '../ModelDraftSubline.svelte';
	import ModelId from '../ModelId.svelte';
	import ModelsManagerStatusCell from './ModelsManagerStatusCell.svelte';
	import { modelRowActions } from './row-actions';
	import { configuredContext, downloadProgressFor, isRowControlClick } from './utils';
	import { Heart, MoreHorizontal } from '@lucide/svelte';
	import { ContextMenuActions, DropdownMenuActions } from '$lib/components/app';
	import { MODEL_ROW_GRID_CLASS, MODEL_ROW_TRAILING_CELL_CLASS } from '$lib/constants';
	import { ModelRowDownloadState } from '$lib/enums';
	import { deviceStore, modelsStore } from '$lib/stores';
	import type { ModelOption, ModelSidecarBadge } from '$lib/types/models';

	interface Props {
		isFavorite: (option: ModelOption) => boolean;
		option: ModelOption;
		onDelete: (option: ModelOption) => void;
		onSelect: (option: ModelOption) => void;
		selected: boolean;
		/** Draft sidecars already on disk, when the row stands for a download. */
		sidecarBadges?: ModelSidecarBadge[];
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let {
		indent = 0,
		isFavorite,
		onDelete,
		onSelect,
		option,
		selected,
		sidecarBadges
	}: Props = $props();

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

	function activate(): void {
		onSelect(option);
	}

	// the "more actions" dropdown and the right-click menu offer the same actions
	let actions = $derived(modelRowActions(option, favorite, isHidden, onDelete, download));
	let contextMenuOpen = $state(false);
</script>

<ContextMenuActions bind:open={contextMenuOpen} {actions}>
	<!-- the whole row opens the model for a pointer; keyboard users reach the same action
     through the model's own button -->
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class={[
			MODEL_ROW_GRID_CLASS,
			'group relative cursor-pointer rounded-md px-2 py-3 transition max-md:px-3 max-md:py-4',
			isHidden && 'opacity-60',
			selected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40',
			contextMenuOpen && !selected && 'bg-muted/40'
		]}
		onclick={(event) => {
			if (!isRowControlClick(event)) activate();
		}}
	>
		<div class="flex min-w-0 flex-col">
			<!-- the row holds a load control and an actions menu, so only the model itself is
	     the button: a button nested in a role="button" row is invalid -->
			<button
				aria-pressed={download ? undefined : selected}
				class="flex min-w-0 cursor-pointer items-center gap-3 text-left max-md:gap-2 max-md:pr-9"
				onclick={activate}
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

					{#if favorite}
						<!-- the heart is decorative, the button's text carries the state -->
						<Heart aria-hidden="true" class="size-icon-sm shrink-0 text-rose-500" />

						<span class="sr-only">favorited</span>
					{/if}
				</span>
			</button>

			{#if sidecarBadges?.length}
				<!-- w-fit keeps the trailing state beside the badges instead of floating
			     to the end of the wide model column -->
				<div class="flex w-fit flex-col pb-1">
					{#each sidecarBadges as badge (badge.repo + ':' + badge.kind + ':' + (badge.quant ?? ''))}
						<ModelDraftSubline
							connectorClass="ml-8"
							kind={badge.kind}
							quant={badge.quant}
							state={ModelRowDownloadState.DOWNLOADED}
						/>
					{/each}
				</div>
			{/if}
		</div>

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

		<div
			class="flex items-center justify-center justify-self-center {MODEL_ROW_TRAILING_CELL_CLASS}"
		>
			<DropdownMenuActions
				{actions}
				align="end"
				triggerIcon={MoreHorizontal}
				triggerTooltip="Model actions"
			/>
		</div>
	</div>
</ContextMenuActions>
