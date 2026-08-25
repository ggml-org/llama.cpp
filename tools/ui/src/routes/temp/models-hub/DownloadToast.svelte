<script lang="ts">
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { ArrowUpRight, Check, TriangleAlert } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { modelsStore } from '$lib/stores';

	interface Props {
		/** `<repo>:<tag>` identifier the server reports progress under. */
		repoWithTag: string;
		/** Repo id (no tag) used by the "Open details" CTA. */
		repoId: string;
		/** Short label for the file being downloaded. */
		displayName: string;
		/** Injected by sonner; dismisses this toast. */
		closeToast?: () => void;
	}

	let { closeToast, displayName, repoId, repoWithTag }: Props = $props();

	// Live progress from the /models/sse feed. The map entry is deleted on
	// download_finished / download_failed, so we latch on first sight to tell
	// "not started yet" apart from "finished".
	let progress = $derived(modelsStore.status.getDownloadProgress(repoWithTag));
	let hasSeenProgress = $state(false);
	let percent = $derived.by(() => {
		if (!progress || progress.totalBytes <= 0) return 0;

		return Math.round((progress.downloadedBytes / progress.totalBytes) * 100);
	});

	$effect(() => {
		if (progress) hasSeenProgress = true;
	});

	// A download is done when the feed drops our entry after we saw progress, or
	// when the model landed in /v1/models / a failure was recorded (covers fast
	// downloads that settle before the first progress event reaches this toast).
	let failed = $derived(modelsStore.status.hasFailedDownload(repoWithTag));
	let modelReady = $derived(modelsStore.status.isModelDownloaded(repoWithTag));
	let done = $derived(
		(hasSeenProgress && !modelsStore.status.isDownloadInProgress(repoWithTag)) ||
			failed ||
			modelReady
	);

	// Auto-dismiss once the download settles; keep the failure visible a bit longer.
	$effect(() => {
		if (!done) return;

		const timer = setTimeout(() => closeToast?.(), failed ? 4000 : 2500);

		return () => clearTimeout(timer);
	});

	function openDetails() {
		closeToast?.();
		goto(`/temp/models-hub/${repoId}`);
	}
</script>

<div class="flex w-full flex-col gap-2">
	<div class="flex items-center justify-between gap-3">
		<span class="flex items-center gap-1.5 text-sm font-medium">
			{#if done && failed}
				<TriangleAlert class="h-4 w-4 text-destructive" />
				Download failed
			{:else if done}
				<Check class="h-4 w-4 text-emerald-500" />
				Download complete
			{:else}
				<span class="h-4 w-4 animate-pulse rounded-full bg-primary/30"></span>
				Downloading
			{/if}
		</span>

		<span class="font-mono text-xs tabular-nums text-muted-foreground">{percent}%</span>
	</div>

	<p class="truncate text-xs text-muted-foreground">{displayName}</p>

	<DownloadProgressBar
		downloadedBytes={progress?.downloadedBytes ?? 0}
		totalBytes={progress?.totalBytes ?? 0}
	/>

	<button
		class="inline-flex items-center justify-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
		onclick={openDetails}
		type="button"
	>
		Open details
		<ArrowUpRight class="h-3.5 w-3.5" />
	</button>
</div>
