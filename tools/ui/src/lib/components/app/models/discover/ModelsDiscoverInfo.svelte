<script lang="ts">
	import ModelId from '../ModelId.svelte';
	import { type DraftVariant } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsHubStore } from '$lib/stores';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import type { ModelModalities } from '$lib/types/models';
	import { detectToolUseSupport, formatParameters } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		model: HfModelInfo;
	}

	let { model }: Props = $props();

	let contextLength = $derived(model.gguf?.context_length);

	// Params badge fallback: the id usually carries the count (`Qwen3.8-27B`),
	// but ids like `Kimi-K3` do not. Fall back to the HF param count
	// (`gguf.total`); search results omit `gguf`, so fetch details lazily only
	// when the name has no params token.
	let fetchedParams = $state<number | null>(null);

	$effect(() => {
		fetchedParams = null;

		if (model.gguf?.total || ModelsService.parseModelId(model.id).params) return;

		let cancelled = false;

		void HuggingFaceService.getDetails(model.id).then((info) => {
			if (!cancelled && info?.gguf?.total) fetchedParams = info.gguf.total;
		});

		return () => {
			cancelled = true;
		};
	});

	let hfParams = $derived(model.gguf?.total ?? fetchedParams);
	let paramsFallback = $derived(
		hfParams && !ModelsService.parseModelId(model.id).params
			? formatParameters(hfParams, 0)
			: undefined
	);

	// Reasoning support from the chat template, matching the details view.
	let supportsThinking = $derived(
		Boolean(model.gguf?.chat_template && /think|reasoning/i.test(model.gguf.chat_template))
	);

	// Tool use support from the chat template.
	let supportsToolUse = $derived(detectToolUseSupport(model.gguf?.chat_template ?? ''));

	// Modalities derived from HF metadata: vision from an mmproj sidecar or a
	// multimodal pipeline tag, audio/video from their pipeline tags.
	let modalities = $derived.by<ModelModalities>(() => {
		const tag = model.pipeline_tag ?? '';
		const vision =
			['image-text-to-text', 'image-to-text', 'text-to-image', 'image-to-video'].includes(tag) ||
			Boolean(model.siblings?.some((s) => s.rfilename.toLowerCase().includes('mmproj')));
		const audio = [
			'audio-classification',
			'audio-to-audio',
			'automatic-speech-recognition',
			'text-to-speech',
			'voice-activity-detection'
		].includes(tag);
		const video = ['text-to-video', 'image-to-video', 'video-to-video'].includes(tag);

		return { audio, video, vision };
	});

	// Draft sidecars (mtp, dflash, dspark, eagle3) present in the repo, e.g.
	// speculative-decoding drafts. mmproj is excluded: it is vision, already
	// conveyed by the modalities.
	let draftVariants = $derived.by<DraftVariant[]>(() => {
		const set = new SvelteSet<DraftVariant>();

		for (const sibling of model.siblings ?? []) {
			const variant = HuggingFaceService.extractQuantMeta(sibling.rfilename)?.variant;

			if (variant && variant !== 'mmproj') set.add(variant);
		}

		return [...set];
	});

	// Combined min/max size: the catalog gives main-model sizes per quant, and
	// the repo file tree carries draft sidecar sizes (the detail siblings do
	// not). Min = smallest main + smallest draft, max = largest main + largest
	// draft, so the stored model fits within the range.
	let sizeRange = $state<{ min: number; max: number } | null>(null);

	$effect(() => {
		const base = modelsHubStore.sizeRangeFor(model.id);

		let cancelled = false;

		if (draftVariants.length === 0) {
			sizeRange = base ?? null;

			return;
		}

		void HuggingFaceService.getTree(model.id).then((tree) => {
			if (cancelled) return;

			const drafts = tree
				.filter((f) => {
					const variant = HuggingFaceService.extractQuantMeta(f.path)?.variant;

					return variant && variant !== 'mmproj';
				})
				.map((f) => f.size ?? 0)
				.filter((size) => size > 0);

			if (base && drafts.length > 0) {
				sizeRange = {
					max: base.max + Math.max(...drafts),
					min: base.min + Math.min(...drafts)
				};
			} else {
				sizeRange = base ?? null;
			}
		});

		return () => {
			cancelled = true;
		};
	});
</script>

<span class="min-w-0 flex-1">
	<ModelId
		modelId={model.id}
		hideOrgName
		{modalities}
		{supportsThinking}
		{supportsToolUse}
		{contextLength}
		{sizeRange}
		{draftVariants}
		params={paramsFallback}
		iconsOnNewLine
		wrap
		class="min-w-0"
	/>
</span>
