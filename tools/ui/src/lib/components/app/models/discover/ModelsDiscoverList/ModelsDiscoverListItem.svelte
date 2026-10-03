<script lang="ts">
	import ModelDownloadProgressBar from '../../ModelDownloadProgressBar.svelte';
	import ModelId from '../../ModelId.svelte';
	import ModelOrgAvatar from '../../ModelOrgAvatar.svelte';
	import { Pause, Play, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import { HF_MMPROJ_FILENAME_TOKEN, HF_MODALITY_PIPELINE_TAGS } from '$lib/constants';
	import { ModelDraftSidecar } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsDiscoverStore, modelsStore } from '$lib/stores';
	import type { ModelsDiscoverSizeRange } from '$lib/stores/models-discover/index.svelte';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import type { ModelDownloadProgress, ModelModalities } from '$lib/types/models';
	import { detectThinkingSupport, detectToolUseSupport, isDraftSidecar, orgOf } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		model: HfModelInfo;
		active?: boolean;
		/** State of a tracked download this row stands for, else the row is a plain result. */
		download?: {
			isPaused: boolean;
			/** Live while the download runs, frozen while it is paused. */
			progress: ModelDownloadProgress | null;
			repoWithTag: string;
			/** Ask the list to confirm cancelling; the list owns the single dialog. */
			onRequestCancel?: (repoWithTag: string) => void;
		};
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let { active = false, download, model, onSelect, showBaseModelAvatar = false }: Props = $props();

	let percent = $derived(
		download?.progress && download.progress.totalBytes > 0
			? Math.round((download.progress.downloadedBytes / download.progress.totalBytes) * 100)
			: null
	);

	let org = $derived(orgOf(model.id));

	// Org whose avatar is shown: the base model's org when showBaseModelAvatar
	// (e.g. the Qwen logo for ggml-org/Qwen3.8-27B-GGUF), else the repo's org.
	let avatarOrg = $derived.by(() => {
		if (!showBaseModelAvatar) return org;

		return orgOf(HuggingFaceService.getBaseModels(model)[0]) || org;
	});

	let contextLength = $derived(model.gguf?.context_length);

	// Reasoning support from the chat template, matching the details view.
	let supportsThinking = $derived(detectThinkingSupport(model.gguf?.chat_template ?? ''));

	// Tool use support from the chat template.
	let supportsToolUse = $derived(detectToolUseSupport(model.gguf?.chat_template ?? ''));

	// Modalities derived from HF metadata: vision from an mmproj sidecar or a
	// multimodal pipeline tag, audio/video from their pipeline tags.
	let modalities = $derived.by<ModelModalities>(() => {
		const tag = model.pipeline_tag ?? '';
		const vision =
			HF_MODALITY_PIPELINE_TAGS.vision.includes(tag) ||
			Boolean(
				model.siblings?.some((s) => s.rfilename.toLowerCase().includes(HF_MMPROJ_FILENAME_TOKEN))
			);
		const audio = HF_MODALITY_PIPELINE_TAGS.audio.includes(tag);
		const video = HF_MODALITY_PIPELINE_TAGS.video.includes(tag);

		return { audio, video, vision };
	});

	// Draft sidecars (mtp, dflash, dspark, eagle3) present in the repo, e.g.
	// speculative-decoding drafts. The list only tells which drafts the repo
	// offers, not how they are configured, so one badge per kind and no quant.
	let draftKinds = $derived.by<ModelDraftSidecar[]>(() => {
		const kinds = new SvelteSet<ModelDraftSidecar>();

		for (const sibling of model.siblings ?? []) {
			const sidecar = HuggingFaceService.extractQuantMeta(sibling.rfilename)?.sidecar;

			if (sidecar && isDraftSidecar(sidecar)) kinds.add(sidecar);
		}

		return Object.values(ModelDraftSidecar).filter((kind) => kinds.has(kind));
	});

	// Min/max size across the repo's quants, draft sidecars included. The store
	// has catalog rows covered already; any other row (a search hit) measures
	// its repo once here and the result is cached per repo.
	let measuredSize = $state<ModelsDiscoverSizeRange | null>(null);
	let sizeRange = $derived(modelsDiscoverStore.cachedSizeRangeFor(model.id) ?? measuredSize);

	$effect(() => {
		const id = model.id;

		if (modelsDiscoverStore.cachedSizeRangeFor(id)) return;

		let cancelled = false;

		void modelsDiscoverStore.sizeRange(id).then((range) => {
			if (!cancelled) measuredSize = range ?? null;
		});

		return () => {
			cancelled = true;
		};
	});
</script>

<li
	class="group relative flex items-center gap-0.5 overflow-hidden rounded-lg transition-colors {active
		? 'bg-primary/10 hover:bg-primary/15'
		: 'hover:bg-muted/60'}"
>
	<button
		aria-current={active ? 'page' : undefined}
		class="flex min-w-0 flex-1 cursor-pointer items-start gap-2.5 p-2.5 text-left"
		onclick={() => onSelect?.(model.id)}
		type="button"
	>
		<ModelOrgAvatar class="mt-1" org={avatarOrg} quantOrg={showBaseModelAvatar ? org : undefined} />

		<span class="min-w-0 flex-1">
			<ModelId
				class="min-w-0"
				{contextLength}
				{draftKinds}
				hideOrgName
				iconsOnNewLine
				{modalities}
				modelId={model.id}
				{sizeRange}
				{supportsThinking}
				{supportsToolUse}
				wrap
			/>
		</span>
	</button>

	{#if download}
		{@const state = download}

		<!-- the percent sits on the row's centre line, and the download's controls take
		     its place on hover, the way the table's status column does -->
		<span class="flex shrink-0 items-center gap-0.5 pr-2.5">
			<span
				class="text-xs text-muted-foreground tabular-nums group-hover:hidden [@media(pointer:coarse)]:hidden"
			>
				{percent !== null ? `${percent}%` : state.isPaused ? 'Paused' : 'Downloading'}
			</span>

			<span class="hidden items-center gap-0.5 group-hover:flex [@media(pointer:coarse)]:flex">
				<ActionIcon
					icon={state.isPaused ? Play : Pause}
					iconSize="h-4 w-4"
					onclick={() =>
						void (state.isPaused
							? modelsStore.status.downloadModel(state.repoWithTag)
							: modelsStore.status.pauseDownload(state.repoWithTag))}
					stopPropagationOnClick
					tooltip={state.isPaused ? 'Resume downloading' : 'Pause downloading'}
					tooltipAsTitle
				/>

				<ActionIcon
					class="text-muted-foreground hover:text-destructive"
					icon={X}
					iconSize="h-4 w-4"
					onclick={() => state.onRequestCancel?.(state.repoWithTag)}
					stopPropagationOnClick
					tooltip="Cancel downloading"
					tooltipAsTitle
				/>
			</span>
		</span>

		{#if state.progress && state.progress.totalBytes > 0}
			<ModelDownloadProgressBar
				downloadedBytes={state.progress.downloadedBytes}
				overlay
				totalBytes={state.progress.totalBytes}
			/>
		{/if}
	{/if}
</li>
