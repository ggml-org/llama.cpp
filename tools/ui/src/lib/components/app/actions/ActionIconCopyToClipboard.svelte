<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
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
	iconSize={ICON_CLASS_DEFAULT}
	disabled={!canCopy}
	onclick={() => canCopy && copyToClipboard(text)}
/>
