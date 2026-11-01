<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Portal } from 'bits-ui';
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
     the thumb (see --corner-control-bottom and --corner-control-side). It lives on the body,
     pinned to the screen: inside the page, the page's slide aside for the sidebar would make
     the page its box and move it. It slides with the page instead, by the same offset and
     ease (see --sidebar-reveal-x in the root layout), and takes no taps while the page is aside -->
<Portal>
	<Button
		aria-label={label}
		class="fixed right-(--corner-control-side) bottom-(--corner-control-bottom) z-20 size-14 translate-x-(--sidebar-reveal-x) rounded-full shadow-lg transition-[translate] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] md:hidden [:root[data-sidebar-reveal-dragging]_&]:transition-none [:root[data-sidebar-revealed]_&]:pointer-events-none {className}"
		{onclick}
	>
		<Icon class="size-6" />
	</Button>
</Portal>
