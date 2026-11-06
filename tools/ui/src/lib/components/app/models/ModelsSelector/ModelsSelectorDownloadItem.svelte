<script lang="ts">
	import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
	import ModelDraftSubline from '../ModelDraftSubline.svelte';
	import ModelOrgAvatar from '../ModelOrgAvatar.svelte';
	import { Info, Loader2, Pause, Play, X } from '@lucide/svelte';
	import { ActionIcon, ModelId } from '$lib/components/app';
	import { ModelRowDownloadState } from '$lib/enums';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore, uiStore } from '$lib/stores';
	import type { ModelDownloadEntry, ModelSidecarBadge } from '$lib/types';

	interface Props {
		/** One entry from the status feed: an in-flight or paused download. */
		entry: ModelDownloadEntry;
		/** Draft sidecars already on disk, rendered as checked sub-lines. */
		sidecars?: ModelSidecarBadge[];
		/** Ask the list to confirm cancelling; the list owns the single dialog. */
		onRequestCancel?: (repoWithTag: string) => void;
		/** Show the organization name in the repo id, as the other selector rows do. */
		showOrgName?: boolean;
	}

	let { entry, onRequestCancel, showOrgName = true, sidecars = [] }: Props = $props();

	/** A draft still coming down badges the id, an on-disk one keeps its sub-line. */
	let pendingDrafts = $derived(sidecars.filter((badge) => badge.pending));
	let onDiskDrafts = $derived(sidecars.filter((badge) => !badge.pending));

	let percent = $derived(
		entry.progress && entry.progress.totalBytes > 0
			? Math.round((entry.progress.downloadedBytes / entry.progress.totalBytes) * 100)
			: null
	);

	let actionText = $derived(entry.isPaused ? 'Resume downloading' : 'Pause downloading');

	// the base model org resolves lazily via HF when unknown
	let orgName = $derived(ModelsService.parseModelId(entry.repoWithTag).orgName);
	let fetchedBaseModelOrg = $state<string | null>(null);

	$effect(() => {
		const repoWithTag = entry.repoWithTag;

		fetchedBaseModelOrg = null;

		// avatars hidden with the Hub metadata setting off: no lookups at all
		if (!HuggingFaceService.isEnabled()) return;

		let cancelled = false;

		void HuggingFaceService.getBaseModel(repoWithTag)
			.then((base) => {
				if (!cancelled && base?.org) fetchedBaseModelOrg = base.org;
			})
			// best-effort lookup: offline or unknown repos keep the repo org
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});
</script>

<!-- One in-flight download: the row opens the model, the status slot beside it
     pauses / resumes and the trailing X cancels (stops and discards the partial
     files). The status slot is always there, the affordances fade in on hover,
     so the list never reflows. The repo's on-disk sidecars hang off the item as
     compact sub-lines instead of rows of their own. -->
<div class="group relative rounded-sm transition-colors hover:bg-accent/50">
	<div
		class="flex items-center gap-2 p-2 text-left text-sm max-md:gap-2.5 max-md:px-2.5 max-md:py-2.5"
	>
		<button
			aria-label="Manage model"
			class="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
			onclick={() => uiStore.openModelDetails(entry.repoWithTag)}
			type="button"
		>
			{#if orgName}
				<ModelOrgAvatar
					org={fetchedBaseModelOrg ?? orgName}
					quantOrg={orgName}
					quantPositionClass="-bottom-1 -right-1"
					quantSize="h-3 w-3"
					size="h-6 w-6"
				/>
			{/if}

			<ModelId
				class="flex-1"
				draftSidecars={pendingDrafts}
				hideOrgName={!showOrgName}
				modelId={entry.repoWithTag}
				showRawTooltip
			/>

			{#if percent !== null}
				<span class="ml-auto shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
					{percent}%
				</span>
			{:else if entry.isPaused}
				<span class="ml-auto shrink-0 text-xs text-muted-foreground">Paused</span>
			{/if}
		</button>

		<!-- status action: spinner -> pause on hover while in flight, play on hover
		     when paused; opacity only for the spinner, the spin owns the transform -->
		<button
			aria-label={actionText}
			class="relative inline-flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center"
			onclick={() => void modelsStore.status.toggleDownload(entry.repoWithTag)}
			type="button"
		>
			{#if entry.isPaused}
				<Play
					class="absolute h-4 w-4 scale-75 opacity-0 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
				/>
			{:else}
				<Loader2
					class="absolute h-4 w-4 animate-spin text-muted-foreground transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
				/>

				<Pause
					class="absolute h-4 w-4 opacity-0 transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
				/>
			{/if}
		</button>

		<!-- the row actions keep the sizing and reveal of the other selector rows -->
		<ActionIcon
			class="pointer-events-none h-5 w-5 shrink-0 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 hover:text-foreground [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
			icon={Info}
			iconSize="h-4 w-4"
			onclick={() => uiStore.openModelDetails(entry.repoWithTag)}
			tooltip="Manage model"
			tooltipAsTitle
		/>

		<ActionIcon
			ariaLabel="Cancel downloading"
			class="pointer-events-none h-5 w-5 shrink-0 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 hover:text-destructive [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
			icon={X}
			iconSize="h-4 w-4"
			onclick={() => onRequestCancel?.(entry.repoWithTag)}
		/>
	</div>

	<!-- only with sub-lines: an empty strip would make the row taller than the
	     plain ones, and the progress bar overlays the row's own bottom edge -->
	{#if onDiskDrafts.length}
		<div class="flex flex-col px-2 pb-1.5">
			{#each onDiskDrafts as badge (badge.repo + ':' + badge.kind + ':' + (badge.quant ?? ''))}
				<ModelDraftSubline
					connectorClass="ml-4"
					kind={badge.kind}
					quant={badge.quant}
					state={ModelRowDownloadState.DOWNLOADED}
				/>
			{/each}
		</div>
	{/if}

	{#if entry.progress && entry.progress.totalBytes > 0}
		<ModelDownloadProgressBar
			downloadedBytes={entry.progress.downloadedBytes}
			overlay
			totalBytes={entry.progress.totalBytes}
		/>
	{/if}
</div>
