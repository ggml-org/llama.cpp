<script lang="ts">
	import ModelCapabilityIcons from '../../ModelCapabilityIcons.svelte';
	import ModelsDiscoverAvatar from '../ModelsDiscoverAvatar.svelte';
	import ModelsDiscoverDetailsMetadata from './ModelsDiscoverDetailsMetadata.svelte';
	import { Download, ExternalLink, Heart } from '@lucide/svelte';
	import { ICON_CLASS_SM } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo, HfModelGguf } from '$lib/types/huggingface';
	import { orgOf } from '$lib/utils';

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
	let repoOrg = $derived(orgOf(details.id) || orgOf(modelId));
	let baseOrg = $derived(orgOf(baseModels[0]));
	let avatarOrg = $derived(baseOrg || repoOrg);
	let quantOrg = $derived(baseOrg && baseOrg !== repoOrg ? repoOrg : undefined);
</script>

<header class="space-y-3">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelsDiscoverAvatar
				org={avatarOrg}
				{quantOrg}
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="h-6 w-6"
				size="h-12 w-12"
			/>

			<div class="min-w-0">
				<div class="flex items-center gap-2">
					<h1 class="truncate text-lg font-semibold">{details.id ?? modelId}</h1>

					<ModelCapabilityIcons
						gapClass="gap-2"
						iconSize="h-4 w-4"
						modalities={{ audio: false, video: false, vision: hasVision }}
						supportsThinking={hasReasoning}
						supportsToolUse={hasTools}
					/>
				</div>

				{#if baseModels.length}
					<div class="flex items-center gap-1">
						<span class="truncate text-xs text-muted-foreground">{baseModels.join(', ')}</span>

						<a
							aria-label="View base model on HuggingFace"
							class="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
							href={HuggingFaceService.getModelUrl(baseModels[0])}
							rel="noopener noreferrer"
							target="_blank"
						>
							<ExternalLink class="h-3 w-3" />
						</a>
					</div>
				{/if}
			</div>
		</div>

		<a
			class="inline-flex shrink-0 items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
			href={HuggingFaceService.getModelUrl(modelId)}
			rel="noopener noreferrer"
			target="_blank"
		>
			<img alt="" class="h-3.5 w-3.5" src="/recommended-mcp/huggingface.ico" />

			View on Hugging Face

			<ExternalLink class={ICON_CLASS_SM} />
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

	<ModelsDiscoverDetailsMetadata {details} {gguf} {licenseTag} {modelId} />
</header>
