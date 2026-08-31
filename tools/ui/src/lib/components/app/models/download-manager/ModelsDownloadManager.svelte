<script lang="ts">
	import { Trash2 } from '@lucide/svelte';
	import DownloadProgressBar from '$lib/components/app/models/discover/DownloadProgressBar.svelte';
	import { modelsStore } from '$lib/stores';

	interface Props {
		open?: boolean;
	}

	let { open = false }: Props = $props();
</script>

{#if open}
	<div class="space-y-2">
		{#each modelsStore.status.downloadEntries() as entry (entry.repoWithTag)}
			<div class="flex flex-col gap-1 rounded-md border p-3">
				<div class="flex items-center justify-between gap-2">
					<span class="truncate font-mono text-xs">{entry.repoWithTag}</span>

					<button
						aria-label="Delete model"
						class="shrink-0 text-muted-foreground/60 transition-colors hover:text-destructive"
						onclick={() => void modelsStore.status.cancelDownload(entry.repoWithTag)}
						type="button"
					>
						<Trash2 class="h-4 w-4" />
					</button>
				</div>

				{#each Object.entries(entry.progress.files) as [file, fileProgress] (file)}
					<div class="space-y-0.5">
						<div class="flex items-center justify-between text-muted-foreground">
							<span class="truncate font-mono text-xs">{file}</span>

							<span class="font-mono tabular-nums">
								{fileProgress.total > 0
									? Math.round((fileProgress.done / fileProgress.total) * 100)
									: 0}%
							</span>
						</div>

						<DownloadProgressBar
							downloadedBytes={fileProgress.done}
							totalBytes={fileProgress.total}
						/>
					</div>
				{/each}
			</div>
		{/each}
	</div>
{/if}
