<script lang="ts">
	import ModelsDiscoverAvatar from './ModelsDiscoverAvatar.svelte';
	import ModelsDiscoverChatTemplateDialog from './ModelsDiscoverChatTemplateDialog.svelte';
	import ModelsDiscoverDetailsName from './ModelsDiscoverDetailsName.svelte';
	import { Download, ExternalLink, Heart, MessageSquareCode } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import { modelsHubStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelGguf } from '$lib/types/huggingface';
	import { formatParameters } from '$lib/utils';
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SM } from '$lib/constants';

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

	let { baseModels, details, gguf, hasReasoning, hasTools, hasVision, licenseTag, modelId }: Props =
		$props();

	// Avatar shows the base model's org (e.g. the Qwen logo for a ggml-org GGUF)
	// with the quant org as a corner badge when they differ.
	let repoOrg = $derived(details.id?.split('/')[0] ?? modelId.split('/')[0] ?? modelId);
	let baseOrg = $derived(baseModels[0]?.split('/')[0]);
	let avatarOrg = $derived(baseOrg || repoOrg);
	let quantOrg = $derived(baseOrg && baseOrg !== repoOrg ? repoOrg : undefined);

	// Catalog family description when curated, else the HF card description.
	let description = $derived(
		modelsHubStore.descriptionFor(modelId) ?? details.cardData?.description
	);

	let chatTemplateOpen = $state(false);
</script>

<header class="space-y-3">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelsDiscoverAvatar org={avatarOrg} {quantOrg} />
			<ModelsDiscoverDetailsName
				modelId={details.id ?? modelId}
				{baseModels}
				{hasVision}
				{hasTools}
				{hasReasoning}
			/>
		</div>

		<a
			href={HuggingFaceService.getModelUrl(modelId)}
			target="_blank"
			rel="noopener noreferrer"
			class="inline-flex shrink-0 items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
		>
			<img src="/recommended-mcp/huggingface.ico" alt="" class="h-3.5 w-3.5" />

			View on Hugging Face

			<ExternalLink class={ICON_CLASS_SM} />
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

	{#if description}
		<p class="text-sm text-muted-foreground">{description}</p>
	{/if}

	<!-- Metadata chips: label | value pairs, matching the HF model page style -->
	<div class="flex flex-wrap items-center gap-1.5">
		{#if gguf?.total}
			<span class="inline-flex items-center divide-x divide-border rounded-md border text-xs">
				<span class="px-2.5 py-1 text-muted-foreground">Model size</span>
				<span class="px-2.5 py-1 font-medium">{formatParameters(gguf.total)} params</span>
			</span>
		{/if}
		{#if gguf?.context_length}
			<span class="inline-flex items-center divide-x divide-border rounded-md border text-xs">
				<span class="px-2.5 py-1 text-muted-foreground">Context</span>
				<span class="px-2.5 py-1 font-medium">{gguf.context_length.toLocaleString()}</span>
			</span>
		{/if}
		{#if gguf?.architecture}
			<span class="inline-flex items-center divide-x divide-border rounded-md border text-xs">
				<span class="px-2.5 py-1 text-muted-foreground">Architecture</span>
				<span class="px-2.5 py-1 font-medium">{gguf.architecture}</span>
			</span>
		{/if}
		{#if gguf?.chat_template}
			<button
				type="button"
				onclick={() => (chatTemplateOpen = true)}
				class="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors hover:bg-muted"
			>
				<MessageSquareCode class="h-3 w-3" />
				Chat template
			</button>
		{/if}
		{#if licenseTag}
    		<span class="inline-flex items-center divide-x divide-border rounded-md border text-xs">
    			<span class="px-2.5 py-1 text-muted-foreground">License</span>
    			<span class="px-2.5 py-1 font-medium">{licenseTag}</span>
    		</span>
			<span class="rounded bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
			</span>
		{/if}
		{#if details.gated === true}
			<span
				class="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400"
			>
				gated
			</span>
		{/if}
	</div>
</header>

{#if gguf?.chat_template}
	<ModelsDiscoverChatTemplateDialog
		bind:open={chatTemplateOpen}
		chatTemplate={gguf.chat_template}
	/>
{/if}
