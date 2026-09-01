<script lang="ts">
	import ModelId from '../ModelId.svelte';
	import ModelsDiscoverAvatar from './ModelsDiscoverAvatar.svelte';
	import { isAuxSidecar, type ModelSidecar } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsHubStore } from '$lib/stores';
	import type { ModelsHubSizeRange } from '$lib/stores/models-hub/index.svelte';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import type { ModelModalities } from '$lib/types/models';
	import { detectThinkingSupport, detectToolUseSupport, formatParameters } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		model: HfModelInfo;
		active?: boolean;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let { active = false, model, onSelect, showBaseModelAvatar = false }: Props = $props();

	let org = $derived(model.id.split('/')[0] ?? model.id);

	// Org whose avatar is shown: the base model's org when showBaseModelAvatar
	// (e.g. the Qwen logo for ggml-org/Qwen3.8-27B-GGUF), else the repo's org.
	let avatarOrg = $derived.by(() => {
		if (!showBaseModelAvatar) return org;

		const base = HuggingFaceService.getBaseModels(model)[0];

		return base?.split('/')[0] || org;
	});

	let contextLength = $derived(model.gguf?.context_length);

	// Params badge fallback: the id usually carries the count (`Qwen3.8-27B`),
	// but ids like `Kimi-K3` do not. Fall back to the HF param count
	// (`gguf.total`), fetched lazily only when neither the response nor the name
	// has it.
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
			? formatParameters(hfParams)
			: undefined
	);

	// Reasoning support from the chat template, matching the details view.
	let supportsThinking = $derived(detectThinkingSupport(model.gguf?.chat_template ?? ''));

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
	let draftSidecars = $derived.by<ModelSidecar[]>(() => {
		const set = new SvelteSet<ModelSidecar>();

		for (const sibling of model.siblings ?? []) {
			const sidecar = HuggingFaceService.extractQuantMeta(sibling.rfilename)?.sidecar;

			if (sidecar && !isAuxSidecar(sidecar)) set.add(sidecar);
		}

		return [...set];
	});

	// Min/max size across the repo's quants, draft sidecars included. The store
	// has catalog rows covered already; any other row (a search hit) measures
	// its repo once here and the result is cached per repo.
	let measuredSize = $state<ModelsHubSizeRange | null>(null);
	let sizeRange = $derived(modelsHubStore.cachedSizeRangeFor(model.id) ?? measuredSize);

	$effect(() => {
		const id = model.id;

		if (modelsHubStore.cachedSizeRangeFor(id)) return;

		let cancelled = false;

		void modelsHubStore.sizeRange(id).then((range) => {
			if (!cancelled) measuredSize = range ?? null;
		});

		return () => {
			cancelled = true;
		};
	});
</script>

<li>
	<button
		aria-current={active ? 'page' : undefined}
		class="flex w-full cursor-pointer items-start gap-2.5 rounded-lg p-2.5 text-left transition-colors {active
			? 'bg-primary/10 hover:bg-primary/15'
			: 'hover:bg-muted/60'}"
		onclick={() => onSelect?.(model.id)}
		type="button"
	>
		<ModelsDiscoverAvatar
			class="mt-1"
			org={avatarOrg}
			quantOrg={showBaseModelAvatar ? org : undefined}
		/>

		<span class="min-w-0 flex-1">
			<ModelId
				class="min-w-0"
				{contextLength}
				{draftSidecars}
				hideOrgName
				iconsOnNewLine
				{modalities}
				modelId={model.id}
				params={paramsFallback}
				{sizeRange}
				{supportsThinking}
				{supportsToolUse}
				wrap
			/>
		</span>
	</button>
</li>
