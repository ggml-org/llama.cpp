<script lang="ts">
	import { Download, HardDriveDownload, Trash2 } from '@lucide/svelte';
	import DownloadProgressBar from '$lib/components/app/models/discover/DownloadProgressBar.svelte';
	import { modelsStore } from '$lib/stores';
	import { ServerModelStatus } from '$lib/enums';

	function isLoaded(status: ServerModelStatus | null): boolean {
		return status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING;
	}
</script>

<div class="space-y-4">
	{#if modelsStore.status.downloadEntries().length}
		<section class="space-y-2">
			<h3
				class="flex items-center gap-1.5 text-xs font-medium tracking-wide text-muted-foreground uppercase"
			>
				<Download class="h-3.5 w-3.5" />
				In progress
			</h3>

			{#each modelsStore.status.downloadEntries() as entry (entry.repoWithTag)}
				<div class="flex flex-col gap-1 rounded-md border p-3">
					<div class="flex items-center justify-between gap-2">
						<span class="truncate font-mono text-xs">{entry.repoWithTag}</span>

						<button
							aria-label="Cancel download"
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
		</section>
	{/if}

	<section class="space-y-2">
		<h3
			class="flex items-center gap-1.5 text-xs font-medium tracking-wide text-muted-foreground uppercase"
		>
			<HardDriveDownload class="h-3.5 w-3.5" />
			Downloaded
		</h3>

		{#if modelsStore.status.downloadedEntries().length}
			{#each modelsStore.status.downloadedEntries() as entry (entry.id)}
				<div class="flex items-center justify-between gap-2 rounded-md border p-3">
					<span class="truncate font-mono text-xs">{entry.id}</span>

					<div class="flex shrink-0 items-center gap-2">
						{#if isLoaded(entry.status)}
							<span
								class="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-primary uppercase"
							>
								Loaded
							</span>
						{/if}

						<button
							aria-label="Delete model"
							class="shrink-0 text-muted-foreground/60 transition-colors hover:text-destructive"
							onclick={() => void modelsStore.status.cancelDownload(entry.id)}
							type="button"
						>
							<Trash2 class="h-4 w-4" />
						</button>
					</div>
				</div>
			{/each}
		{:else}
			<p class="text-sm text-muted-foreground">No downloaded models yet.</p>
		{/if}
	</section>
</div>
