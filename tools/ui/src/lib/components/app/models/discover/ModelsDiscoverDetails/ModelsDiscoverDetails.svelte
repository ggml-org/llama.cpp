<script lang="ts">
	import ModelsDiscoverDetailsHeader from './ModelsDiscoverDetailsHeader.svelte';
	import ModelsDiscoverDetailsReadme from './ModelsDiscoverDetailsReadme.svelte';
	import ModelsDiscoverDetailsSkeleton from './ModelsDiscoverDetailsSkeleton.svelte';
	import { ModelAuxSidecar } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types';
	import { detectThinkingSupport, detectToolUseSupport } from '$lib/utils';

	interface Props {
		/** Full HuggingFace model id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** Model details from `/api/models/{id}?full=true`; null while loading. */
		details: HfModelDetailInfo | null;
		/** GGUF files of the repo, shards collapsed, sorted by size desc. */
		files: HfModelSibling[];
		/** README.md content, frontmatter stripped; null when unavailable. */
		readme: string | null;
		/** True while the model data is being fetched. */
		loading?: boolean;
		/** Error message when loading failed. */
		error?: string | null;
	}

	let { details, error = null, files, loading = false, modelId, readme }: Props = $props();

	let gguf = $derived(details?.gguf);
	let baseModels = $derived(HuggingFaceService.getBaseModels(details));
	let licenseTag = $derived.by(() => {
		const tags = details?.tags ?? [];

		return tags.find((t) => t.startsWith('license:'))?.replace('license:', '') ?? null;
	});

	// Capabilities derived from HF metadata. Vision comes from an mmproj sidecar
	// or a multimodal pipeline tag; tool use / reasoning from the chat template.
	let hasMmproj = $derived(
		files.some(
			(f) => HuggingFaceService.extractQuantMeta(f.path)?.sidecar === ModelAuxSidecar.MMPROJ
		)
	);
	let hasVision = $derived(hasMmproj || details?.pipeline_tag === 'image-text-to-text');
	let hasTools = $derived(detectToolUseSupport(gguf?.chat_template ?? ''));
	let hasReasoning = $derived(detectThinkingSupport(gguf?.chat_template ?? ''));
</script>

{#if loading}
	<ModelsDiscoverDetailsSkeleton />
{:else if error}
	<div class="flex h-full items-center justify-center py-20">
		<p class="text-sm text-destructive">{error}</p>
	</div>
{:else if details}
	<div class="space-y-6 p-6">
		<ModelsDiscoverDetailsHeader
			{baseModels}
			{details}
			{gguf}
			{hasReasoning}
			{hasTools}
			{hasVision}
			{licenseTag}
			{modelId}
		/>

		<ModelsDiscoverDetailsReadme {readme} />
	</div>
{/if}
