<script lang="ts">
	import ActionIcon from './ActionIcon.svelte';
	import { Copy } from '@lucide/svelte';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';
	import { copyToClipboard } from '#lib/utils/index.ts';

	interface Props {
		ariaLabel?: string;
		canCopy?: boolean;
		text: string;
	}

	let { ariaLabel = 'Copy to clipboard', canCopy = true, text }: Props = $props();
</script>

<ActionIcon
	disabled={!canCopy}
	icon={Copy}
	iconSize={ICON_CLASS_DEFAULT}
	onclick={() => canCopy && copyToClipboard(text)}
	tooltip={ariaLabel}
/>
