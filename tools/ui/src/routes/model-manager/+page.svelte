<script lang="ts">
	import {
		Copy,
		ExternalLink,
		Heart,
		HeartOff,
		LoaderCircle,
		Plus,
		Power,
		PowerOff,
		Trash2,
		X
	} from '@lucide/svelte';
	import { ActionIcon, DialogModelsDiscover, ModelId } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, serverStore } from '$lib/stores';
	import type { ApiModelDataEntry } from '$lib/types';
	import type { ModelOption } from '$lib/types/models';
	import { buildModelManagerTree, copyToClipboard, getRepoId, resolveBaseModel } from '$lib/utils';
	import { onMount } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';

	let modelsHubOpen = $state(false);
	let baseModels = new SvelteMap<string, string | null>();

	let isRouter = $derived(serverStore.isRouterMode);

	let tree = $derived(buildModelManagerTree(modelsStore.models, baseModels));

	onMount(() => {
		void modelsStore.fetch();
		modelsStore.status.subscribe();
	});

	// Resolve the original model for each installed repo (cached, best-effort).
	$effect(() => {
		const repoIds = new Set(modelsStore.models.map((m) => getRepoId(m.model)));

		for (const repoId of repoIds) {
			void resolveBaseModel(repoId).then((base) => {
				baseModels.set(repoId, base);
			});
		}
	});

	function getRouterEntry(modelId: string): ApiModelDataEntry | undefined {
		return modelsStore.routerModels.find((m) => m.id === modelId);
	}

	function hfUrl(modelId: string): string | null {
		const repoId = getRepoId(modelId);

		if (!repoId.includes('/')) return null;

		return `https://huggingface.co/${repoId}`;
	}
</script>

{#snippet modelRow(option: ModelOption, indent: boolean, hideQuantization: boolean)}
	{@const entry = getRouterEntry(option.model)}
	{@const status = entry?.status}
	{@const statusValue = status?.value}
	{@const isLoaded =
		statusValue === ServerModelStatus.LOADED || statusValue === ServerModelStatus.SLEEPING}
	{@const isLoading = statusValue === ServerModelStatus.LOADING}
	{@const isDownloading = statusValue === ServerModelStatus.DOWNLOADING}
	{@const isFailed = statusValue === ServerModelStatus.FAILED || status?.failed === true}
	{@const canRemove = entry?.can_remove === true}
	{@const isFav = modelsStore.isFavorite(option.model)}
	{@const hfLink = hfUrl(option.model)}
	{@const downloadProgress = modelsStore.status.getDownloadProgress(option.model)}

	<tr class="border-t transition-colors hover:bg-muted/40">
		<td class="px-3 py-2 {indent ? 'pl-8' : ''}">
			<ModelId
				modelId={option.model}
				aliases={option.aliases}
				tags={option.tags}
				modalities={option.modalities}
				supportsThinking={modelsStore.props.checkModelSupportsThinking(option.model)}
				{hideQuantization}
				showRawTooltip
			/>
		</td>

		<td class="px-3 py-2">
			<span class="inline-flex items-center gap-1.5 text-xs">
				{#if isLoading}
					<LoaderCircle class="h-3.5 w-3.5 animate-spin text-muted-foreground" />
					<span>Loading</span>
				{:else if isDownloading}
					<LoaderCircle class="h-3.5 w-3.5 animate-spin text-muted-foreground" />
					<span>Downloading</span>
				{:else if isLoaded}
					<span class="h-2 w-2 rounded-full bg-green-500"></span>
					<span>{statusValue === ServerModelStatus.SLEEPING ? 'Sleeping' : 'Loaded'}</span>
				{:else if isFailed}
					<span class="h-2 w-2 rounded-full bg-red-500"></span>
					<span>Failed</span>
				{:else if statusValue === ServerModelStatus.DOWNLOADED}
					<span class="h-2 w-2 rounded-full bg-muted-foreground/50"></span>
					<span>Downloaded</span>
				{:else}
					<span class="h-2 w-2 rounded-full bg-muted-foreground/50"></span>
					<span>Unloaded</span>
				{/if}
			</span>

			{#if isDownloading && downloadProgress}
				<div class="mt-1 h-1 w-32 overflow-hidden rounded-full bg-muted">
					<div
						class="h-full bg-primary"
						style="width: {downloadProgress.totalBytes > 0
							? Math.round((downloadProgress.downloadedBytes / downloadProgress.totalBytes) * 100)
							: 0}%"
					></div>
				</div>
			{/if}
		</td>

		<td class="px-3 py-2">
			<div class="flex items-center justify-end gap-0.5">
				{#if isRouter}
					{#if isLoading}
						<ActionIcon
							icon={X}
							tooltip="Cancel load"
							onclick={() => modelsStore.status.cancelLoad(option.model)}
						/>
					{:else if isDownloading}
						<ActionIcon
							icon={X}
							tooltip="Cancel download"
							onclick={() => modelsStore.status.cancelDownload(option.model)}
						/>
					{:else if isLoaded}
						<ActionIcon
							icon={PowerOff}
							tooltip="Unload model"
							onclick={() => modelsStore.status.unload(option.model)}
						/>
					{:else}
						<ActionIcon
							icon={Power}
							tooltip="Load model"
							onclick={() => modelsStore.status.load(option.model)}
						/>
					{/if}

					{#if canRemove && !isDownloading}
						<ActionIcon
							icon={Trash2}
							tooltip="Delete model"
							onclick={() => modelsStore.status.cancelDownload(option.model)}
						/>
					{/if}
				{/if}

				<ActionIcon
					icon={isFav ? HeartOff : Heart}
					tooltip={isFav ? 'Remove from favorites' : 'Add to favorites'}
					onclick={() => modelsStore.toggleFavorite(option.model)}
				/>

				<ActionIcon
					icon={Copy}
					tooltip="Copy model id"
					onclick={() => copyToClipboard(option.model)}
				/>

				{#if hfLink}
					<a
						href={hfLink}
						target="_blank"
						rel="noreferrer"
						class="inline-flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted-foreground/10 hover:text-foreground"
						aria-label="View on HuggingFace"
						title="View on HuggingFace"
					>
						<ExternalLink class="h-3.5 w-3.5" />
					</a>
				{/if}
			</div>
		</td>
	</tr>
{/snippet}

<div class="mx-auto flex h-full w-full max-w-5xl flex-col gap-4 p-4 md:p-6">
	<header class="flex items-center justify-between gap-2">
		<div>
			<h1 class="text-lg font-semibold">Model Manager</h1>

			<p class="text-sm text-muted-foreground">
				Models installed on this machine and available through /v1/models.
			</p>
		</div>

		<Button size="sm" onclick={() => (modelsHubOpen = true)}>
			<Plus class="h-4 w-4" />

			Add more models
		</Button>
	</header>

	{#if modelsStore.loading}
		<div class="flex items-center justify-center py-16">
			<p class="text-sm text-muted-foreground">Loading models...</p>
		</div>
	{:else if modelsStore.error}
		<div class="rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
			<p class="text-sm text-destructive">{modelsStore.error}</p>
		</div>
	{:else if modelsStore.models.length === 0}
		<div class="flex flex-col items-center justify-center gap-3 py-16 text-center">
			<p class="text-sm text-muted-foreground">No models installed.</p>

			<Button size="sm" variant="outline" onclick={() => (modelsHubOpen = true)}>
				<Plus class="h-4 w-4" />

				Add more models
			</Button>
		</div>
	{:else}
		<div class="overflow-hidden rounded-lg border">
			<table class="w-full text-sm">
				<thead class="bg-muted/40 text-left text-xs text-muted-foreground">
					<tr>
						<th class="px-3 py-2 font-medium">Model</th>
						<th class="px-3 py-2 font-medium">Status</th>
						<th class="px-3 py-2 text-right font-medium">Actions</th>
					</tr>
				</thead>

				<tbody>
					{#each tree as parent (parent.parentId)}
						<tr class="border-t bg-muted/20">
							<td colspan="3" class="px-3 py-1.5 text-xs font-semibold text-muted-foreground">
								{parent.parentId}
							</td>
						</tr>

						{#each parent.quantOrgs as org (org.repoId)}
							<tr class="border-t bg-muted/10">
								<td colspan="3" class="px-6 py-1.5 text-xs font-medium text-muted-foreground">
									{org.repoId}
								</td>
							</tr>

							{#each org.quants as quant (quant.quant ?? 'default')}
								{@render modelRow(quant.main, false, false)}

								{#each quant.drafts as draft (draft.option.id)}
									{@render modelRow(draft.option, true, true)}
								{/each}
							{/each}
						{/each}
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<DialogModelsDiscover bind:open={modelsHubOpen} />
