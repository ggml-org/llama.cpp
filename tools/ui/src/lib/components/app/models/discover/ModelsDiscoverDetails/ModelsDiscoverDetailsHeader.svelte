<script lang="ts">
	import ModelCapabilityIcons from '../../ModelCapabilityIcons.svelte';
	import ModelOrgAvatar from '../../ModelOrgAvatar.svelte';
	import ModelsDiscoverDetailsHfHubStats from './ModelsDiscoverDetailsHfHubStats.svelte';
	import ModelsDiscoverDetailsMetadata from './ModelsDiscoverDetailsMetadata.svelte';
	import { ExternalLink } from '@lucide/svelte';
	import { base } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
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
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelOrgAvatar
				org={avatarOrg}
				{quantOrg}
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="size-6"
				size="size-12"
			/>

			<div class="min-w-0">
				<div class="flex items-center gap-2 @max-lg:flex-col @max-lg:items-start">
					<h1 class="truncate text-lg font-semibold @max-lg:w-full">{details.id ?? modelId}</h1>

					<ModelCapabilityIcons
						gapClass="gap-2"
						iconSize="size-icon-md"
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
							<ExternalLink class="size-icon-xs" />
						</a>
					</div>
				{/if}
			</div>
		</div>

		<Button
			class="h-auto gap-1.5 px-2.5 py-1.5 text-xs"
			href={HuggingFaceService.getModelUrl(modelId)}
			rel="noopener noreferrer"
			size="sm"
			target="_blank"
			variant="secondary"
		>
			<img alt="" class="size-3.5" src="{base}/recommended-mcp/huggingface.ico" />

			View on Hugging Face

			<ExternalLink class="size-3.5" />
		</Button>
	</div>

	<ModelsDiscoverDetailsHfHubStats {details} />

	<ModelsDiscoverDetailsMetadata {details} {gguf} {licenseTag} {modelId} />
</header>
