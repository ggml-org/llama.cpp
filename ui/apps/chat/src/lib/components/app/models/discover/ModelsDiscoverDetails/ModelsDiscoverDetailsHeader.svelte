<script lang="ts">
	import ModelOrgAvatar from '../../ModelOrgAvatar.svelte';
	import ModelsDiscoverDetailsHfHubStats from './ModelsDiscoverDetailsHfHubStats.svelte';
	import ModelsDiscoverDetailsMetadata from './ModelsDiscoverDetailsMetadata.svelte';
	import { ModelCapabilityIcons } from '@ggml-org/llama-ui/models';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { ExternalLink } from '@lucide/svelte';
	import { HuggingFaceService } from '#lib/services/index.ts';
	import type { HfModelDetailInfo, HfModelGguf } from '#lib/types/huggingface.js';
	import type { ModelModalities } from '#lib/types/models.js';
	import { orgOf } from '#lib/utils/index.ts';
	import { asset } from '$app/paths';

	interface Props {
		modelId: string;
		details: HfModelDetailInfo;
		gguf?: HfModelGguf;
		baseModels: string[];
		licenseTag: string | null;
		modalities: ModelModalities;
		hasTools: boolean;
		hasReasoning: boolean;
	}

	let {
		baseModels,
		details,
		gguf,
		hasReasoning,
		hasTools,
		licenseTag,
		modalities,
		modelId
	}: Props = $props();

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
					<h1 class="truncate text-title @max-lg:w-full">{details.id ?? modelId}</h1>

					<ModelCapabilityIcons
						gapClass="gap-2"
						iconSize="size-icon-md"
						{modalities}
						supportsThinking={hasReasoning}
						supportsToolUse={hasTools}
					/>
				</div>

				{#if baseModels.length}
					<div class="flex items-center gap-1">
						<span class="truncate text-caption text-muted-foreground">{baseModels.join(', ')}</span>

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
			class="h-auto gap-1.5 px-2.5 py-1.5 text-caption"
			href={HuggingFaceService.getModelUrl(modelId)}
			rel="noopener noreferrer"
			size="sm"
			target="_blank"
			variant="secondary"
		>
			<img alt="" class="size-3.5" src={asset('recommended-mcp/huggingface.ico')} />

			View on Hugging Face

			<ExternalLink class="size-3.5" />
		</Button>
	</div>

	<ModelsDiscoverDetailsHfHubStats {details} />

	<ModelsDiscoverDetailsMetadata {details} {gguf} {licenseTag} {modelId} />
</header>
