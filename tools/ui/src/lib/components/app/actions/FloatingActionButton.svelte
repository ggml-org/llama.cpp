<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import type { Component } from 'svelte';

	interface Props {
		class?: string;
		icon: Component;
		/** What it does, read out in place of a visible label. */
		label: string;
		onclick?: () => void;
	}

	let { class: className = '', icon: Icon, label, onclick }: Props = $props();
</script>

<!-- a phone's main action of a page: a round key in the bottom-right corner, within reach of
     the thumb, clear of the home indicator; held in place while the page slides aside -->
<Button
	aria-label={label}
	class="sidebar-reveal-pin-bottom fixed right-4 bottom-safe-6 z-20 size-14 rounded-full shadow-lg md:hidden {className}"
	{onclick}
>
	<Icon class="size-6" />
</Button>
