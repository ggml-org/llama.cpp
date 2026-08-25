<script lang="ts">
	import ModelsDiscoverDetailsDownloadOptions from './ModelsDiscoverDetailsDownloadOptions.svelte';
	import ModelsDiscoverDetailsHeader from './ModelsDiscoverDetailsHeader.svelte';
	import ModelsDiscoverDetailsReadme from './ModelsDiscoverDetailsReadme.svelte';
	import TerminalCommands from './TerminalCommands.svelte';
	import { type DraftVariant } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types/huggingface';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Full HuggingFace model id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
	}

	let { modelId }: Props = $props();

	let details = $state<HfModelDetailInfo | null>(null);
	let files = $state<HfModelSibling[]>([]);
	let readme = $state<string | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	let gguf = $derived(details?.gguf);
	let baseModels = $derived(HuggingFaceService.getBaseModels(details));
	let licenseTag = $derived.by(() => {
		const tags = details?.tags ?? [];

		return tags.find((t) => t.startsWith('license:'))?.replace('license:', '') ?? null;
	});

	// Capabilities derived from HF metadata. Vision comes from an mmproj sidecar
	// or a multimodal pipeline tag; tool use / reasoning from the chat template.
	let hasMmproj = $derived(
		files.some((f) => HuggingFaceService.extractQuantMeta(f.path)?.variant === 'mmproj')
	);
	let hasVision = $derived(hasMmproj || details?.pipeline_tag === 'image-text-to-text');
	let hasTools = $derived(Boolean(gguf?.chat_template && /tools?[_\s}]/i.test(gguf.chat_template)));
	let hasReasoning = $derived(
		Boolean(gguf?.chat_template && /think|reasoning/i.test(gguf.chat_template))
	);

	// Draft sidecar variants (mtp, dflash, dspark, eagle3) present in the repo,
	// e.g. speculative-decoding drafts. mmproj is excluded: it is vision.
	let draftVariants = $derived.by<DraftVariant[]>(() => {
		const set = new SvelteSet<DraftVariant>();

		for (const file of files) {
			const variant = HuggingFaceService.extractQuantMeta(file.path)?.variant;

			if (variant && variant !== 'mmproj') set.add(variant);
		}

		return [...set];
	});

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };
	let bitDepthRows = $derived.by<BitDepthRow[]>(() => {
		const rows = new SvelteMap<number, HfModelSibling[]>();

		for (const file of files) {
			const meta = HuggingFaceService.extractQuantMeta(file.path);

			// mmproj sidecars are already conveyed by the Vision capability badge.
			if (meta?.variant === 'mmproj') continue;

			const depth = meta?.quant ? HuggingFaceService.getBitDepth(meta.quant) : null;
			const bucket = depth ?? 99;
			const list = rows.get(bucket) ?? [];

			list.push(file);
			rows.set(bucket, list);
		}

		return Array.from(rows.entries())
			.map(([bitDepth, rowFiles]) => ({ bitDepth, files: rowFiles }))
			.sort((a, b) => a.bitDepth - b.bitDepth);
	});

	async function load(id: string) {
		loading = true;
		error = null;

		try {
			const [info, tree, readmeText] = await Promise.all([
				HuggingFaceService.getDetails(id),
				HuggingFaceService.getTree(id),
				HuggingFaceService.getReadme(id)
			]);

			if (!info) {
				error = 'Model not found';

				return;
			}

			details = info;
			files = HuggingFaceService.filterByExtension(tree, '.gguf');
			readme = readmeText;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load model';
		} finally {
			loading = false;
		}
	}

	// Re-fetch when the dialog selects a different model (component is reused).
	$effect(() => {
		void load(modelId);
	});
</script>

{#if loading}
	<div class="flex h-full items-center justify-center py-20">
		<p class="text-sm text-muted-foreground">Loading model...</p>
	</div>
{:else if error}
	<div class="flex h-full items-center justify-center py-20">
		<p class="text-sm text-destructive">{error}</p>
	</div>
{:else if details}
	<div class="space-y-6 p-6">
		<ModelsDiscoverDetailsHeader
			{modelId}
			{details}
			{gguf}
			{baseModels}
			{licenseTag}
			{hasVision}
			{hasTools}
			{hasReasoning}
		/>

		<ModelsDiscoverDetailsDownloadOptions {modelId} {bitDepthRows} />

		<TerminalCommands {modelId} {draftVariants} />

		<ModelsDiscoverDetailsReadme {readme} />
	</div>
{/if}
