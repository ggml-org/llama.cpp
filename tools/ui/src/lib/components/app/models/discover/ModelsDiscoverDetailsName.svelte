<script lang="ts">
	import { ExternalLink, Image, Lightbulb, Wrench } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { HuggingFaceService } from '$lib/services';

	interface Props {
		/** Full HuggingFace model id (quant org + name), e.g. `ggml-org/Qwen3.8-27B-GGUF`. */
		modelId: string;
		/** Base model ids, shown as small text under the quant name. */
		baseModels: string[];
		hasVision: boolean;
		hasTools: boolean;
		hasReasoning: boolean;
	}

	let { baseModels, hasReasoning, hasTools, hasVision, modelId }: Props = $props();
</script>

<div class="min-w-0">
	<div class="flex items-center gap-2">
		<h1 class="truncate text-lg font-semibold">{modelId}</h1>

		{#if hasVision || hasTools || hasReasoning}
			<div class="flex shrink-0 items-center gap-2.5 text-muted-foreground">
				{#if hasVision}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Image class="h-4 w-4" />
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>Vision</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if hasTools}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Wrench class="h-4 w-4" />
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>Tool use</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if hasReasoning}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Lightbulb class="h-4 w-4" />
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>Reasoning</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			</div>
		{/if}
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
