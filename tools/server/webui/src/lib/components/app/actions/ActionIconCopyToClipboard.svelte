<script lang="ts">
	import { Copy } from '@lucide/svelte';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		ariaLabel?: string;
		canCopy?: boolean;
		text: string;
	}

	let { ariaLabel = 'Copy to clipboard', canCopy = true, text }: Props = $props();
</script>

<Copy
	class="h-3 w-3 flex-shrink-0 cursor-{canCopy ? 'pointer' : 'not-allowed'}"
	aria-label={ariaLabel}
	onclick={() => canCopy && copyToClipboard(text)}
/>
