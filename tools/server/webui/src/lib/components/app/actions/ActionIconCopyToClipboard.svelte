<script lang="ts">
	import { Copy } from '@lucide/svelte';
	import { copyToClipboard } from '$lib/utils';
	import ActionIcon from './ActionIcon.svelte';

	export let ariaLabel: string = 'Copy to clipboard';
	export let canCopy: boolean = true;
	export let text: string;
</script>

<ActionIcon
	icon={Copy}
	tooltip={ariaLabel}
	iconSize="h-4 w-4"
	disabled={!canCopy}
	onclick={() => canCopy && copyToClipboard(text)}
/>
