<script lang="ts">
	import { config } from '$lib/stores/settings.svelte';
	import { fade } from 'svelte/transition';
	import { SETTINGS_KEYS } from '$lib/constants';
	import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';

	interface Props {
		modelLoadingText: string | null;
		processingState: UseProcessingStateReturn;
		position: 'top' | 'bottom';
	}

	let { modelLoadingText, processingState, position }: Props = $props();

	const isFullWidth = $derived(Boolean(config()[SETTINGS_KEYS.FULL_WIDTH_CHAT]));
	const marginClass = position === 'top' ? 'mt-6' : 'mt-4';
</script>

<div class="{marginClass} w-full {isFullWidth ? '' : 'max-w-3xl'}" in:fade>
	<div class="flex flex-col items-start gap-2">
		<span class="shimmer-text text-sm">
			{modelLoadingText ??
				processingState.getPromptProgressText() ??
				processingState.getProcessingMessage() ??
				'Processing...'}
		</span>
	</div>
</div>
