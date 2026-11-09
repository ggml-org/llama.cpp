<!--
	The frame of the Architecture pages: the Philosophy pages' Editorial, with the pages' own variables pointed at
	Llama UI (architecture.css), and their links between pages (/stack/, /chat/) opening the matching story.
-->
<script lang="ts">
	import './architecture.css';
	import Editorial from '../philosophy/Editorial.svelte';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();

	let root: HTMLElement | undefined = $state();

	// page paths, as Storybook story ids
	const STORIES: Record<string, string> = {
		'/channels/': 'architecture-channels--channels',
		'/chat/': 'architecture-chat--chat',
		'/stack/': 'architecture-stack--stack'
	};

	// diagrams draw links in SVG too, whose href is an attribute rather than a string property
	function retarget(a: Element) {
		const id = STORIES[(a.getAttribute('href') ?? '').split('#')[0]];

		if (!id) return;

		// the manager page, or this page's folder when the story is opened on its own (iframe.html)
		const manager = (window.top ?? window).location;
		const base = manager.pathname.replace(/iframe\.html$/, '');

		a.setAttribute('href', `${manager.origin}${base}?path=/story/${id}`);
		a.setAttribute('target', '_top');
	}

	$effect(() => {
		root?.querySelectorAll('a[href^="/"]').forEach(retarget);
	});
</script>

<div bind:this={root} class="architecture">
	<Editorial>{@render children()}</Editorial>
</div>
