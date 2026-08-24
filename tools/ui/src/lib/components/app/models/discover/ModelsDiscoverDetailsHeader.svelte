<script lang="ts">
	import { Download, ExternalLink, Heart, Image, Lightbulb, Wrench } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo, HfModelGguf } from '$lib/types/huggingface';
	import ModelsDiscoverAvatar from './ModelsDiscoverAvatar.svelte';
	import ModelsDiscoverDetailsName from './ModelsDiscoverDetailsName.svelte';

	interface Props {
		modelId: string;
		details: HfModelDetailInfo;
		gguf?: HfModelGguf;
		baseModels: string[];
		licenseTag: string | null;
		hasVision: boolean;
		hasTools: boolean;
		hasReasoning: boolean;
	}

	let {
		baseModels,
		details,
		gguf,
		hasReasoning,
		hasTools,
		hasVision,
		licenseTag,
		modelId
	}: Props = $props();

	// Avatar shows the base model's org (e.g. the Qwen logo for a ggml-org GGUF)
	// with the quant org as a corner badge when they differ.
	let repoOrg = $derived(details.id?.split('/')[0] ?? modelId.split('/')[0] ?? modelId);
	let baseOrg = $derived(baseModels[0]?.split('/')[0]);
	let avatarOrg = $derived(baseOrg || repoOrg);
	let quantOrg = $derived(baseOrg && baseOrg !== repoOrg ? repoOrg : undefined);
</script>

<header class="space-y-3">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelsDiscoverAvatar org={avatarOrg} quantOrg={quantOrg} />
			<ModelsDiscoverDetailsName modelId={details.id ?? modelId} {baseModels} />
		</div>
		<a
			href={HuggingFaceService.getModelUrl(modelId)}
			target="_blank"
			rel="noopener noreferrer"
			class="inline-flex shrink-0 items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
		>
			<ExternalLink class="h-3.5 w-3.5" />
			View on HF
		</a>
	</div>

	<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
		{#if typeof details.downloads === 'number'}
			<span class="inline-flex items-center gap-1.5">
				<Download class="h-3.5 w-3.5" />
				{HuggingFaceService.formatDownloads(details.downloads)}
			</span>
		{/if}
		{#if typeof details.likes === 'number'}
			<span class="inline-flex items-center gap-1.5">
				<Heart class="h-3.5 w-3.5" />
				{HuggingFaceService.formatLikes(details.likes)}
			</span>
		{/if}
		{#if details.lastModified}
			<span>Updated {HuggingFaceService.formatRelativeTime(details.lastModified)}</span>
		{/if}
	</div>

	{#if details.cardData?.description}
		<p class="text-sm text-muted-foreground">{details.cardData.description}</p>
	{/if}

	<!-- Metadata chips -->
	<div class="flex flex-wrap items-center gap-1.5">
		{#if gguf?.total}
			<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
				{HuggingFaceService.formatFileSize(gguf.total).replace(' B', '')}B params
			</span>
		{/if}
		{#if gguf?.architecture}
			<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
				{gguf.architecture.replace(/_/g, ' ')}
			</span>
		{/if}
		{#if gguf?.context_length}
			<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
				{gguf.context_length.toLocaleString()} ctx
			</span>
		{/if}
		{#if licenseTag}
			<span class="rounded bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
				{licenseTag}
			</span>
		{/if}
		{#if details.gated === true}
			<span class="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400">
				gated
			</span>
		{/if}
	</div>

	<!-- Capability badges -->
	{#if hasVision || hasTools || hasReasoning}
		<div class="flex flex-wrap items-center gap-1.5">
			{#if hasVision}
				<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
					<Image class="h-3 w-3" />
					Vision
				</span>
			{/if}
			{#if hasTools}
				<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
					<Wrench class="h-3 w-3" />
					Tool use
				</span>
			{/if}
			{#if hasReasoning}
				<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
					<Lightbulb class="h-3 w-3" />
					Reasoning
				</span>
			{/if}
		</div>
	{/if}
</header>
