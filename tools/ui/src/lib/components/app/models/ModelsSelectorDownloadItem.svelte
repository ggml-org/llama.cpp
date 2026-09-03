<script lang="ts">
	import DownloadProgressBar from './discover/DownloadProgressBar.svelte';
	import ModelsDiscoverAvatar from './discover/ModelsDiscoverAvatar.svelte';
	import { Loader2, Pause, Play, X } from '@lucide/svelte';
	import { ModelId } from '$lib/components/app';
	import DialogConfirmation from '$lib/components/app/dialogs/DialogConfirmation.svelte';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { ModelDownloadProgress } from '$lib/types';

	interface Props {
		/** One entry from the status feed: an in-flight or paused download. */
		entry: { isPaused: boolean; progress: ModelDownloadProgress | null; repoWithTag: string };
	}

	let { entry }: Props = $props();

	// cancel confirmation state
	let confirmCancelOpen = $state(false);

	let percent = $derived(
		entry.progress && entry.progress.totalBytes > 0
			? Math.round((entry.progress.downloadedBytes / entry.progress.totalBytes) * 100)
			: null
	);

	let actionText = $derived(entry.isPaused ? 'Resume downloading' : 'Pause downloading');

	// Avatar: the repo org with the quantizer org corner badge, as in the model
	// option rows; the base model org resolves lazily via HF when unknown
	let orgName = $derived(ModelsService.parseModelId(entry.repoWithTag).orgName);
	let fetchedBaseModelOrg = $state<string | null>(null);

	$effect(() => {
		const repoWithTag = entry.repoWithTag;

		fetchedBaseModelOrg = null;

		let cancelled = false;

		void HuggingFaceService.getBaseModel(repoWithTag).then((base) => {
			if (!cancelled && base?.org) fetchedBaseModelOrg = base.org;
		});

		return () => {
			cancelled = true;
		};
	});
</script>

<!-- One in-flight download; same actions as the discover quant chips: the row
     itself pauses / resumes, the trailing X cancels (stops and discards the
     partial files). Both affordances fade in on hover, the slots are reserved
     so the list never reflows. -->
<div
	class="group relative flex items-center gap-2 rounded-sm p-2 text-left text-sm transition-colors hover:bg-accent/50"
>
	<button
		aria-label={actionText}
		class="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
		onclick={() => {
			if (entry.isPaused) void modelsStore.status.downloadModel(entry.repoWithTag).catch(() => {});
			else void modelsStore.status.pauseDownload(entry.repoWithTag);
		}}
		type="button"
	>
		{#if orgName}
			<ModelsDiscoverAvatar
				class="mt-0"
				org={fetchedBaseModelOrg ?? orgName}
				quantOrg={orgName}
				quantPositionClass="-bottom-1 -right-1"
				quantSize="h-3 w-3"
				size="h-6 w-6"
			/>
		{/if}

		<ModelId class="flex-1" hideOrgName modelId={entry.repoWithTag} showRawTooltip />

		{#if percent !== null}
			<span class="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">{percent}%</span>
		{:else if entry.isPaused}
			<span class="shrink-0 text-xs text-muted-foreground">Paused</span>
		{/if}

		<!-- status action: spinner -> pause on hover while in flight, play on hover
				 when paused; opacity only for the spinner, the spin owns the transform -->
		<span class="relative inline-flex h-4 w-4 shrink-0 items-center justify-center">
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
		</span>
	</button>

	<button
		aria-label="Cancel downloading"
		class="inline-flex h-4 w-4 shrink-0 scale-75 cursor-pointer items-center justify-center rounded-sm text-muted-foreground/70 opacity-0 transition-[opacity,transform,color] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] hover:text-destructive group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
		onclick={() => (confirmCancelOpen = true)}
		type="button"
	>
		<X class="h-4 w-4" />
	</button>

	{#if entry.progress && entry.progress.totalBytes > 0}
		<DownloadProgressBar
			downloadedBytes={entry.progress.downloadedBytes}
			overlay
			totalBytes={entry.progress.totalBytes}
		/>
	{/if}
</div>

<DialogConfirmation
	cancelText="Keep downloading"
	confirmText="Cancel download"
	description={`This stops the download of ${modelsStore.toDisplayName(entry.repoWithTag)} and removes the partial files. Pause it instead to keep the progress.`}
	onCancel={() => (confirmCancelOpen = false)}
	onConfirm={() => {
		confirmCancelOpen = false;

		void modelsStore.status.cancelDownload(entry.repoWithTag);
	}}
	open={confirmCancelOpen}
	title="Cancel download"
	variant="destructive"
/>
