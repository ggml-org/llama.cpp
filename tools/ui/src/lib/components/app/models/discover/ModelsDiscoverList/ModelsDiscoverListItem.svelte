<script lang="ts">
	import ModelDownloadProgressBar from '../../ModelDownloadProgressBar.svelte';
	import ModelDraftSubline from '../../ModelDraftSubline.svelte';
	import ModelId from '../../ModelId.svelte';
	import ModelOrgAvatar from '../../ModelOrgAvatar.svelte';
	import { Pause, Play, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import { HF_MMPROJ_FILENAME_TOKEN, HF_MODALITY_PIPELINE_TAGS, MODEL_ID } from '$lib/constants';
	import { ModelDraftSidecar, ModelRowDownloadState } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsDiscoverStore, modelsStore } from '$lib/stores';
	import type { ModelsDiscoverSizeRange } from '$lib/stores/models-discover/index.svelte';
	import type { HfModelInfo, HfModelSiblingRef } from '$lib/types/huggingface';
	import type {
		ModelDownloadProgress,
		ModelModalities,
		ModelSidecarBadge
	} from '$lib/types/models';
	import {
		detectThinkingSupport,
		detectToolUseSupport,
		directoryOf,
		isDraftSidecar,
		isServerSidecarFile,
		orgOf
	} from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	/**
	 * True when a file of the weights sits in the draft's directory: the server
	 * resolves a sidecar against the model file beside it, never across directories.
	 */
	function hasWeightsBeside(siblings: HfModelSiblingRef[], draftPath: string): boolean {
		const directory = directoryOf(draftPath);

		return siblings.some(
			(sibling) =>
				MODEL_ID.WEIGHT_EXTENSION_REGEX.test(sibling.rfilename) &&
				directoryOf(sibling.rfilename) === directory &&
				!HuggingFaceService.extractQuantMeta(sibling.rfilename)?.sidecar
		);
	}

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
		/** Draft sidecars already on disk, rendered as checked sub-lines. */
		sidecarBadges?: ModelSidecarBadge[];
		onSelect?: (modelId: string) => void;
	}

	let {
		active = false,
		download,
		model,
		onSelect,
		showBaseModelAvatar = false,
		sidecarBadges = []
	}: Props = $props();

	/** A draft still coming down badges the id, an on-disk one keeps its sub-line. */
	let pendingDrafts = $derived(sidecarBadges.filter((badge) => badge.pending));
	let onDiskDrafts = $derived(sidecarBadges.filter((badge) => !badge.pending));

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

	// Draft sidecars (mtp, dflash, dspark, eagle3) the repo offers, badged with no
	// quant: the list does not know how the server configures them. Only the drafts
	// the server can actually pull are shown, so a badge never outlives the file
	// find_best_sibling would resolve.
	let draftKinds = $derived.by<ModelDraftSidecar[]>(() => {
		const siblings = model.siblings ?? [];
		const kinds = new SvelteSet<ModelDraftSidecar>();

		for (const sibling of siblings) {
			const sidecar = HuggingFaceService.extractQuantMeta(sibling.rfilename)?.sidecar;

			if (!sidecar || !isDraftSidecar(sidecar)) continue;

			if (!isServerSidecarFile(sibling.rfilename, sidecar)) continue;

			if (!hasWeightsBeside(siblings, sibling.rfilename)) continue;

			kinds.add(sidecar);
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
	class="group relative overflow-hidden rounded-lg transition-colors {active
		? 'bg-primary/10 hover:bg-primary/15'
		: 'hover:bg-muted/60'}"
>
	<div class="flex items-center gap-0.5">
		<button
			aria-current={active ? 'page' : undefined}
			class="flex min-w-0 flex-1 cursor-pointer items-start gap-2.5 p-2.5 text-left"
			onclick={() => onSelect?.(model.id)}
			type="button"
		>
			<ModelOrgAvatar
				class="mt-1"
				org={avatarOrg}
				quantOrg={showBaseModelAvatar ? org : undefined}
			/>

			<span class="min-w-0 flex-1">
				<!-- a download row names what is downloading: the tag's quant and sidecar
			     badges, not the repo's whole sidecar offering -->
				<ModelId
					class="min-w-0"
					{contextLength}
					draftKinds={download ? undefined : draftKinds}
					draftSidecars={download ? pendingDrafts : undefined}
					hideOrgName
					iconsOnNewLine
					{modalities}
					modelId={download ? download.repoWithTag : model.id}
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
						iconSize="size-icon-md"
						onclick={() => void modelsStore.status.toggleDownload(state.repoWithTag)}
						stopPropagationOnClick
						tooltip={state.isPaused ? 'Resume downloading' : 'Pause downloading'}
						tooltipAsTitle
					/>

					<ActionIcon
						class="text-muted-foreground hover:text-destructive"
						icon={X}
						iconSize="size-icon-md"
						onclick={() => state.onRequestCancel?.(state.repoWithTag)}
						stopPropagationOnClick
						tooltip="Cancel downloading"
						tooltipAsTitle
					/>
				</span>
			</span>
		{/if}
	</div>

	<!-- only with sub-lines, so a plain row is the row's own height and the
	     progress bar overlays its bottom edge -->
	{#if onDiskDrafts.length}
		<div class="flex flex-col px-2.5 pb-1.5">
			{#each onDiskDrafts as badge (badge.repo + ':' + badge.kind + ':' + (badge.quant ?? ''))}
				<ModelDraftSubline
					connectorClass="ml-8"
					kind={badge.kind}
					quant={badge.quant}
					state={ModelRowDownloadState.DOWNLOADED}
				/>
			{/each}
		</div>
	{/if}

	{#if download?.progress && download.progress.totalBytes > 0}
		<ModelDownloadProgressBar
			downloadedBytes={download.progress.downloadedBytes}
			overlay
			totalBytes={download.progress.totalBytes}
		/>
	{/if}
</li>
