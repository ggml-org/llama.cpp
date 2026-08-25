<script lang="ts">
	import ModelsDiscoverAvatar from './ModelsDiscoverAvatar.svelte';
	import ModelsDiscoverChatTemplateDialog from './ModelsDiscoverChatTemplateDialog.svelte';
	import ModelsDiscoverDetailsName from './ModelsDiscoverDetailsName.svelte';
	import {
		Download,
		ExternalLink,
		Heart,
		Image,
		Lightbulb,
		MessageSquareCode,
		Wrench
	} from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { HuggingFaceService } from '$lib/services';
	import { modelsHubStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelGguf } from '$lib/types/huggingface';
	import { formatParameters } from '$lib/utils';

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

	{#if description}
		<p class="text-sm text-muted-foreground">{description}</p>
	{/if}

	<!-- Metadata chips -->
	<div class="flex flex-wrap items-center gap-1.5">
		{#if gguf?.total}
			<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
				{formatParameters(gguf.total)} params
			</span>
		{/if}
		{#if gguf?.architecture}
			<span
				class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground capitalize"
			>
				{gguf.architecture.replace(/_/g, ' ')}
			</span>
		{/if}
		{#if gguf?.context_length}
			<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
				{gguf.context_length.toLocaleString()} ctx
			</span>
		{/if}
		{#if gguf?.chat_template}
			<button
				type="button"
				onclick={() => (chatTemplateOpen = true)}
				class="inline-flex items-center gap-1 rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground transition-colors hover:bg-secondary/70"
			>
				<MessageSquareCode class="h-3 w-3" />
				Chat template
			</button>
		{/if}
		{#if licenseTag}
			<span class="rounded bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
				{licenseTag}
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

	<!-- Capability / modality icons, matching the list item's icon style -->
	{#if hasVision || hasTools || hasReasoning}
		<div class="flex flex-wrap items-center gap-2.5 text-muted-foreground">
			{#if hasVision}
				<Tooltip.Root>
					<Tooltip.Trigger>
						<Image class="h-4 w-4" />
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>Vision</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}
			{#if hasTools}
				<Tooltip.Root>
					<Tooltip.Trigger>
						<Wrench class="h-4 w-4" />
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>Tool use</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}
			{#if hasReasoning}
				<Tooltip.Root>
					<Tooltip.Trigger>
						<Lightbulb class="h-4 w-4" />
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>Reasoning</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}
		</div>
	{/if}
</header>

{#if gguf?.chat_template}
	<ModelsDiscoverChatTemplateDialog
		bind:open={chatTemplateOpen}
		chatTemplate={gguf.chat_template}
	/>
{/if}
