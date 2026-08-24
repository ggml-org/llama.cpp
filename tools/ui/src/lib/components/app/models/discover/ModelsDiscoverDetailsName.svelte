<script lang="ts">
	import { Copy, ExternalLink } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		/** Full HuggingFace model id (quant org + name), e.g. `ggml-org/Qwen3.8-27B-GGUF`. */
		modelId: string;
		/** Base model ids, shown as small text under the quant name. */
		baseModels: string[];
	}

	let { baseModels, modelId }: Props = $props();
</script>

<div class="min-w-0">
	<div class="flex items-center gap-2">
		<h1 class="truncate text-lg font-semibold">{modelId}</h1>
		<button
			type="button"
			onclick={() => copyToClipboard(modelId)}
			class="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
			aria-label="Copy model id"
		>
			<Copy class="h-4 w-4" />
		</button>
	</div>

	{#if baseModels.length}
		<div class="flex items-center gap-1">
			<span class="truncate text-xs text-muted-foreground">{baseModels.join(', ')}</span>
			<a
				href={HuggingFaceService.getModelUrl(baseModels[0])}
				target="_blank"
				rel="noopener noreferrer"
				class="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
				aria-label="View base model on HuggingFace"
			>
				<ExternalLink class="h-3 w-3" />
			</a>
		</div>
	{/if}
</div>
