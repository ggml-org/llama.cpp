<script lang="ts">
	import ActionIcon from './ActionIcon.svelte';
	import { Check, Copy } from '@lucide/svelte';
	import { COPY_FEEDBACK_RESET_MS, ICON_CLASS_DEFAULT } from '$lib/constants';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		ariaLabel?: string;
		canCopy?: boolean;
		text: string;
	}

	let { ariaLabel = 'Copy to clipboard', canCopy = true, text }: Props = $props();

	// after a copy the icon turns into a green check for a moment, then back
	let isCopied = $state(false);
	// the swap pops in only after the first copy, never on the initial render
	let hasSwapped = $state(false);
	let resetTimer: ReturnType<typeof setTimeout> | undefined;

	let iconClass = $derived(
		[
			ICON_CLASS_DEFAULT,
			isCopied && 'text-fg-success',
			hasSwapped && 'animate-in fade-in zoom-in-75 duration-150 motion-reduce:animate-none'
		]
			.filter(Boolean)
			.join(' ')
	);

	async function copy() {
		if (!canCopy || !(await copyToClipboard(text))) return;

		clearTimeout(resetTimer);
		isCopied = true;
		hasSwapped = true;
		resetTimer = setTimeout(() => (isCopied = false), COPY_FEEDBACK_RESET_MS);
	}

	$effect(() => () => clearTimeout(resetTimer));
</script>

<ActionIcon
	disabled={!canCopy}
	icon={isCopied ? Check : Copy}
	iconSize={iconClass}
	onclick={copy}
	tooltip={ariaLabel}
/>
