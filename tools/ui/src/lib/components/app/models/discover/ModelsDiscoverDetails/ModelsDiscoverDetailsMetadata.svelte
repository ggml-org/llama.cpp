<script lang="ts">
	import ModelsDiscoverChatTemplateDialog from './ModelsDiscoverChatTemplateDialog.svelte';
	import ModelsDiscoverDetailsMetadataItem from './ModelsDiscoverDetailsMetadataItem.svelte';
	import { MessageSquareCode } from '@lucide/svelte';
	import { modelsDiscoverStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelGguf } from '$lib/types/huggingface';
	import { formatParameters } from '$lib/utils';

	interface Props {
		/** Full HuggingFace model id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		details: HfModelDetailInfo;
		gguf?: HfModelGguf;
		licenseTag: string | null;
	}

	let { details, gguf, licenseTag, modelId }: Props = $props();

	// Catalog family description when curated, else the HF card description.
	let description = $derived(
		modelsDiscoverStore.descriptionFor(modelId) ?? details.cardData?.description
	);

	let chatTemplateOpen = $state(false);
</script>

{#if description}
	<p class="text-sm text-muted-foreground">{description}</p>
{/if}

<!-- Metadata chips: label | value pairs, matching the HF model page style -->
<div class="flex flex-wrap items-center gap-1.5">
	{#if gguf?.total}
		<ModelsDiscoverDetailsMetadataItem
			label="Model size"
			value="{formatParameters(gguf.total)} params"
		/>
	{/if}

	{#if gguf?.context_length}
		<ModelsDiscoverDetailsMetadataItem
			label="Context"
			value={gguf.context_length.toLocaleString()}
		/>
	{/if}

	{#if gguf?.architecture}
		<ModelsDiscoverDetailsMetadataItem label="Architecture" value={gguf.architecture} />
	{/if}

	{#if gguf?.chat_template}
		<button
			class="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors hover:bg-muted"
			onclick={() => (chatTemplateOpen = true)}
			type="button"
		>
			<MessageSquareCode class="h-3 w-3" />
			Chat template
		</button>
	{/if}

	{#if licenseTag}
		<ModelsDiscoverDetailsMetadataItem label="License" value={licenseTag} />
	{/if}

	{#if details.gated === true}
		<span
			class="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400"
		>
			gated
		</span>
	{/if}
</div>

{#if gguf?.chat_template}
	<ModelsDiscoverChatTemplateDialog
		bind:open={chatTemplateOpen}
		chatTemplate={gguf.chat_template}
	/>
{/if}
