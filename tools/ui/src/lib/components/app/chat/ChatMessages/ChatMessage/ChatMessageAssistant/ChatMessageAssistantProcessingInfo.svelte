<script lang="ts">
	import { fade } from 'svelte/transition';
	import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';

	interface Props {
		modelLoadingText: string | null;
		processingState: UseProcessingStateReturn;
		position: 'top' | 'bottom';
	}

	let { modelLoadingText, processingState, position }: Props = $props();

	const marginClass = position === 'top' ? 'mt-6' : 'mt-4';
</script>

<div class="{marginClass} w-full max-w-3xl" in:fade>
	<div class="flex flex-col items-start gap-2">
		<span class="shimmer-text text-sm">
			{modelLoadingText ??
				processingState.getPromptProgressText() ??
				processingState.getProcessingMessage() ??
				'Processing...'}
		</span>
	</div>
</div>
