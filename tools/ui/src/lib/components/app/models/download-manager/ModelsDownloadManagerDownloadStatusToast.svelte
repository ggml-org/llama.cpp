<script lang="ts">
	import { X } from '@lucide/svelte';
	import DownloadProgressBar from '$lib/components/app/models/discover/DownloadProgressBar.svelte';
	import { modelsStore } from '$lib/stores';

	interface Props {
		/** `<repo>:<tag>` the download was queued under; progress is read live from the store. */
		repoWithTag: string;
		/** Injected by svelte-sonner; dismisses this toast. */
		closeToast?: () => void;
	}

	let { closeToast, repoWithTag }: Props = $props();

	// Read live from the /models/sse feed, so the toast updates itself.
	let progress = $derived(modelsStore.status.getDownloadProgress(repoWithTag));

	function percent(done: number, total: number): number {
		return total > 0 ? Math.round((done / total) * 100) : 0;
	}

	// The feed clears progress once the download settles; close then - the
	// store's finished/failed toast reports the outcome.
	let hadProgress = $state(false);

	$effect(() => {
		if (progress) {
			hadProgress = true;
		} else if (hadProgress) {
			closeToast?.();
		}
	});
</script>

<div class="w-80 space-y-2 rounded-md border bg-background p-3 shadow-sm">
	<div class="flex items-center justify-between gap-2">
		<span class="truncate text-xs font-medium" title={repoWithTag}>{repoWithTag}</span>

		<button
			aria-label="Dismiss"
			class="shrink-0 text-muted-foreground/60 transition-colors hover:text-foreground"
			onclick={() => closeToast?.()}
			type="button"
		>
			<X class="h-3.5 w-3.5" />
		</button>
	</div>

	{#if progress}
		<div class="space-y-1.5">
			{#each Object.entries(progress.files) as [file, fileProgress] (file)}
				<div class="space-y-0.5">
					<div class="flex items-center justify-between gap-2 text-muted-foreground">
						<span class="truncate font-mono text-xs">{file}</span>

						<span class="shrink-0 font-mono tabular-nums">
							{percent(fileProgress.done, fileProgress.total)}%
						</span>
					</div>

					<DownloadProgressBar
						downloadedBytes={fileProgress.done}
						totalBytes={fileProgress.total}
					/>
				</div>
			{/each}
		</div>
	{:else}
		<p class="text-xs text-muted-foreground">Waiting for the server to report progress...</p>
	{/if}
</div>
