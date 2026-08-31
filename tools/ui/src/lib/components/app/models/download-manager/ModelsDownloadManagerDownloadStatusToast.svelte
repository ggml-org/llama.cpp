<script lang="ts">
	import { X } from '@lucide/svelte';
	import DownloadProgressBar from '$lib/components/app/models/discover/DownloadProgressBar.svelte';
	import type { ModelDownloadProgress } from '$lib/types';

	interface Props {
		/** HuggingFace repo id of the download. */
		repoId: string;
		/** Live progress from the /models/sse feed (per-file). */
		progress: ModelDownloadProgress;
		/** CTA fired to open the download manager dialog. */
		onOpenManager?: () => void;
		/** Dismiss the toast (does not cancel the download). */
		onDismiss?: () => void;
	}

	let { onDismiss, onOpenManager, progress, repoId }: Props = $props();

	let files = $derived(Object.entries(progress.files));

	function percent(done: number, total: number): number {
		return total > 0 ? Math.round((done / total) * 100) : 0;
	}
</script>

<div class="w-80 space-y-2 rounded-md border bg-background p-3 shadow-sm">
	<div class="flex items-center justify-between gap-2">
		<span class="truncate text-xs font-medium" title={repoId}>{repoId}</span>

		<button
			aria-label="Dismiss"
			class="shrink-0 text-muted-foreground/60 transition-colors hover:text-foreground"
			onclick={() => onDismiss?.()}
			type="button"
		>
			<X class="h-3.5 w-3.5" />
		</button>
	</div>

	<div class="space-y-1.5">
		{#each files as [file, fileProgress] (file)}
			<div class="space-y-0.5">
				<div class="flex items-center justify-between gap-2 text-muted-foreground">
					<span class="truncate font-mono text-xs">{file}</span>

					<span class="shrink-0 font-mono tabular-nums">
						{percent(fileProgress.done, fileProgress.total)}%
					</span>
				</div>

				<DownloadProgressBar downloadedBytes={fileProgress.done} totalBytes={fileProgress.total} />
			</div>
		{/each}
	</div>

	{#if onOpenManager}
		<button
			class="w-full rounded-md border px-2 py-1 text-xs font-medium transition-colors hover:bg-muted"
			onclick={() => onOpenManager?.()}
			type="button"
		>
			Open download manager
		</button>
	{/if}
</div>
