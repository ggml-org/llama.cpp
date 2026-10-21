<script lang="ts">
	import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';
	import { fade } from 'svelte/transition';

	interface Props {
		modelLoadingText: string | null;
		processingState: UseProcessingStateReturn;
		/** `inline` drops the chat-turn margins for callers that own the layout. */
		position: 'top' | 'bottom' | 'inline';
	}

	let { modelLoadingText, position, processingState }: Props = $props();

	const wrapperClass = $derived.by(() => {
		if (position === 'top') return 'mt-6 w-full max-w-3xl';

		if (position === 'bottom') return 'mt-4 w-full max-w-3xl';

		return 'w-full';
	});
</script>

<div in:fade class={wrapperClass}>
	<div class="flex flex-col items-start gap-2">
		<span class="shimmer-text text-sm">
			{modelLoadingText ??
				processingState.getPromptProgressText() ??
				processingState.getProcessingMessage() ??
				'Processing...'}
		</span>
	</div>
</div>
