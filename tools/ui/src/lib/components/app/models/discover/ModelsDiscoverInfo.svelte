<script lang="ts">
	import { Download, Heart } from '@lucide/svelte';
	import { type DraftVariant } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import { formatParameters } from '$lib/utils';
	import type { ModelModalities } from '$lib/types/models';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import ModelId from '../ModelId.svelte';

	interface Props {
		model: HfModelInfo;
	}

	let { model }: Props = $props();

	let contextLength = $derived(model.gguf?.context_length);

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

		return { vision, audio, video };
	});

	// Draft sidecars (mtp, dflash, dspark, eagle3) present in the repo, e.g.
	// speculative-decoding drafts. mmproj is excluded: it is vision, already
	// conveyed by the modalities.
	let draftVariants = $derived.by<DraftVariant[]>(() => {
		const set = new Set<DraftVariant>();

		for (const sibling of model.siblings ?? []) {
			const variant = HuggingFaceService.extractQuantMeta(sibling.rfilename)?.variant;

			if (variant && variant !== 'mmproj') set.add(variant);
		}

		return [...set];
	});
</script>

<span class="min-w-0 flex-1">
	<ModelId modelId={model.id} hideOrgName {modalities} {draftVariants} class="min-w-0" />

	<span class="mt-0.5 block truncate text-xs text-muted-foreground">
		<span class="inline-flex items-center gap-1">
			<Download class="h-3 w-3" />
			{HuggingFaceService.formatDownloads(model.downloads)}
		</span>
		<span class="ml-2.5 inline-flex items-center gap-1">
			<Heart class="h-3 w-3" />
			{HuggingFaceService.formatLikes(model.likes)}
		</span>
		{#if contextLength}
			<span class="ml-2.5 inline-flex items-center gap-1">
				{formatParameters(contextLength)} ctx
			</span>
		{/if}
	</span>
</span>
