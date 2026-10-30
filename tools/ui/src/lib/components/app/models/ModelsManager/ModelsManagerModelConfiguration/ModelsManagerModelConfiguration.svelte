<script lang="ts">
	import ModelDownloadProgressBar from '../../ModelDownloadProgressBar.svelte';
	import { downloadProgressFor } from '../utils';
	import ModelsManagerModelConfigurationActions from './ModelsManagerModelConfigurationActions.svelte';
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import { ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { deviceStore, modelsStore, uiStore } from '$lib/stores';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';
	import { repoOf } from '$lib/utils';

	interface Props {
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInChat: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
	}

	let { onClose, onToggleLoad, onUseInChat, onUseInNewChat, option }: Props = $props();

	// the same removal the table row offers, confirmed by the same dialog
	let deleteOpen = $state(false);

	// the props cache is a plain Map, so the read has to name its version to stay reactive
	let serverProps = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelProps(option.model);
	});
	let status = $derived(modelsStore.getModelStatus(option.model));
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));

	// an in-flight download has no configuration: the pane shows its progress and
	// the download actions, everything a loaded or downloaded model offers waits
	let isDownload = $derived(
		modelsStore.status.isDownloadInProgress(option.model) ||
			modelsStore.status.isDownloadPaused(option.model)
	);
	let isDownloadPaused = $derived(modelsStore.status.isDownloadPaused(option.model));
	let downloadProgress = $derived(downloadProgressFor(option.model));
	let downloadPercent = $derived(
		downloadProgress && downloadProgress.totalBytes > 0
			? Math.round((downloadProgress.downloadedBytes / downloadProgress.totalBytes) * 100)
			: null
	);

	// the same cancellation the table's rows confirm, owned by this pane
	let cancelDownloadOpen = $state(false);

	// The server reports the full metadata only once a model is loaded, which loads it.
	// With the Hub enabled, it fills those gaps instead.
	let hubDetails = $state<HfModelDetailInfo | null>(null);

	$effect(() => {
		const repo = repoOf(option.model);

		let cancelled = false;

		hubDetails = null;

		if (!HuggingFaceService.isEnabled() || !repo.includes('/')) {
			return;
		}

		void HuggingFaceService.getDetails(repo)
			.then((details) => {
				if (!cancelled) hubDetails = details;
			})
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});
</script>

<div class="flex h-full min-h-0 flex-col">
	<ModelsManagerModelConfigurationHeader
		{isDownload}
		{isDownloadPaused}
		{isLoaded}
		{onClose}
		onRequestCancelDownload={() => (cancelDownloadOpen = true)}
		onSeeInDiscover={isDownload
			? () => uiStore.openModelsDiscover(repoOf(option.model))
			: undefined}
		onToggleDownload={() => void modelsStore.status.toggleDownload(option.model)}
		{onToggleLoad}
		{onUseInChat}
		{onUseInNewChat}
		{option}
		{status}
	/>

	<div class="min-h-0 flex-1 overflow-y-auto p-4">
		{#if isDownload}
			<!-- nothing is on disk yet, so the pane keeps to the download itself -->
			<div class="space-y-2 py-1">
				<div class="flex items-baseline justify-between gap-3 text-sm">
					<span class="text-muted-foreground">
						{isDownloadPaused ? 'Paused at' : 'Downloaded'}
					</span>

					<span class="font-mono text-xs tabular-nums text-muted-foreground">
						{#if downloadPercent !== null}
							{HuggingFaceService.formatFileSize(downloadProgress?.downloadedBytes ?? 0)}
							/
							{HuggingFaceService.formatFileSize(downloadProgress?.totalBytes ?? 0)}
							({downloadPercent}%)
						{:else}
							Waiting for progress...
						{/if}
					</span>
				</div>

				{#if downloadProgress && downloadProgress.totalBytes > 0}
					<ModelDownloadProgressBar
						downloadedBytes={downloadProgress.downloadedBytes}
						totalBytes={downloadProgress.totalBytes}
					/>
				{/if}
			</div>
		{:else}
			<ModelsManagerModelConfigurationInformation hub={hubDetails} {option} {serverProps} />

			<div class="mt-4 border-t border-border/30 pt-3 pb-2">
				<Button
					class="w-full justify-start gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive dark:hover:bg-destructive/20"
					onclick={() => (deleteOpen = true)}
					size="sm"
					variant="ghost"
				>
					<Trash2 class="size-icon-md" />
					Delete this model from disk
				</Button>
			</div>
		{/if}
	</div>

	<!-- a phone keeps the actions at the bottom of its screen, in reach of the thumb -->
	{#if deviceStore.isMobile}
		<div class="shrink-0 border-t border-border/40 px-4 py-3">
			<ModelsManagerModelConfigurationActions
				{isDownload}
				{isDownloadPaused}
				{isLoaded}
				isLoading={status === ServerModelStatus.LOADING}
				onRequestCancelDownload={() => (cancelDownloadOpen = true)}
				onSeeInDiscover={isDownload
					? () => uiStore.openModelsDiscover(repoOf(option.model))
					: undefined}
				onToggleDownload={() => void modelsStore.status.toggleDownload(option.model)}
				{onToggleLoad}
				{onUseInChat}
				{onUseInNewChat}
			/>
		</div>
	{/if}

	<DialogConfirmDownload
		action={ModelDownloadConfirmAction.DELETE}
		onClose={() => (deleteOpen = false)}
		open={deleteOpen}
		repoWithTag={option.model}
	/>

	<DialogConfirmDownload
		action={ModelDownloadConfirmAction.CANCEL}
		onClose={() => (cancelDownloadOpen = false)}
		onConfirm={() => void modelsStore.status.cancelDownload(option.model)}
		open={cancelDownloadOpen}
		repoWithTag={option.model}
	/>
</div>
