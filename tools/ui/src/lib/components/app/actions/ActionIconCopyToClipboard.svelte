<script lang="ts">
	import ActionIcon from './ActionIcon.svelte';
	import { Copy } from '@lucide/svelte';
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { copyToClipboard } from '$lib/utils';

	export let ariaLabel: string = 'Copy to clipboard';
	export let canCopy: boolean = true;
	export let text: string;
</script>

<ActionIcon
	icon={Copy}
	tooltip={ariaLabel}
	iconSize={ICON_CLASS_DEFAULT}
	disabled={!canCopy}
	onclick={() => canCopy && copyToClipboard(text)}
/>
