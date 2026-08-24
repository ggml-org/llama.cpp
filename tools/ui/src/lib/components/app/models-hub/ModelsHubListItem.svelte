<script lang="ts">
	import { Download, Eye, Heart } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelInfo } from '$lib/types/huggingface';

	interface Props {
		model: HfModelInfo;
		active?: boolean;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let { model, active = false, showBaseModelAvatar = false, onSelect }: Props = $props();

	let org = $derived(model.id.split('/')[0] ?? model.id);
	let name = $derived(model.id.split('/')[1] ?? model.id);
	let params = $derived(HuggingFaceService.parseParamCount(model.id));
	let avatarError = $state(false);

	// Org whose avatar is shown: the base model's org when showBaseModelAvatar
	// (e.g. the Qwen logo for ggml-org/Qwen3.8-27B-GGUF), else the repo's org.
	let avatarOrg = $derived.by(() => {
		if (!showBaseModelAvatar) return org;

		const base = HuggingFaceService.getBaseModels(model)[0];

		return base?.split('/')[0] || org;
	});
	// Vision = a multimodal pipeline tag or an mmproj projector in the repo.
	let hasVision = $derived(
		model.pipeline_tag === 'image-text-to-text' ||
			Boolean(model.siblings?.some((s) => s.rfilename.toLowerCase().includes('mmproj')))
	);

	// Monogram fallback: avatar org initial on a hue derived from its name, so
	// each org gets a stable distinct color.
	let hue = $derived.by(() => {
		let h = 0;

		for (let i = 0; i < avatarOrg.length; i++) h = (h * 31 + avatarOrg.charCodeAt(i)) >>> 0;

		return h % 360;
	});
</script>

<li>
	<button
		type="button"
		onclick={() => onSelect?.(model.id)}
		aria-current={active ? 'page' : undefined}
		class="flex w-full cursor-pointer items-start gap-2.5 rounded-lg p-2.5 text-left transition-colors {active
			? 'bg-primary/10 hover:bg-primary/15'
			: 'hover:bg-muted/60'}"
	>
		{#if avatarError}
			<span
				class="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-sm font-semibold text-white"
				style="background-color: hsl({hue} 60% 45%)"
				aria-hidden="true"
			>
				{avatarOrg.charAt(0).toUpperCase()}
			</span>
		{:else}
			<img
				src={HuggingFaceService.getAvatarUrl(avatarOrg)}
				onerror={() => (avatarError = true)}
				class="mt-0.5 h-9 w-9 shrink-0 rounded-md bg-gray-200 p-0.5"
				alt=""
				loading="lazy"
			/>
		{/if}

		<span class="min-w-0 flex-1">
			<span class="flex items-center justify-between gap-2">
				<span class="truncate text-sm font-medium">{name}</span>
				{#if hasVision}
					<Eye class="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-label="Vision" />
				{/if}
			</span>

			<span class="mt-0.5 block truncate text-xs text-muted-foreground">
				{org}
				{#if model.lastModified}
					· {HuggingFaceService.formatRelativeTime(model.lastModified)}
				{/if}
			</span>

			<span class="mt-1 flex items-center gap-2.5 text-xs text-muted-foreground">
				{#if params}
					<span class="font-medium tabular-nums">{params}</span>
				{/if}
				<span class="flex items-center gap-1">
					<Download class="h-3 w-3" />
					{HuggingFaceService.formatDownloads(model.downloads)}
				</span>
				<span class="flex items-center gap-1">
					<Heart class="h-3 w-3" />
					{HuggingFaceService.formatLikes(model.likes)}
				</span>
			</span>
		</span>
	</button>
</li>
