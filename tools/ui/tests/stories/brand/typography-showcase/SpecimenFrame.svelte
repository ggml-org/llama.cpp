<!--
	The type specimen's frame: the page it sits on, set in the version and weight on show, with the V shortcut for the
	next version. A copy of the page template in ggml-org/llama.brand (fonts/specimen/src/lib/templates/Doc.svelte);
	the specimen in lib/ and TypographyShowcase.svx are copied from there too, so change them there first.
	variant and weight pin a story to one version and weight; left out, they follow the page's own controls.
-->
<script lang="ts">
	import '../fonts/fonts.css';
	import './lib/styles/specimen.css';
	import { fallback, specimen } from './lib/specimen';
	import { view } from './lib/view.svelte';
	import type { Snippet } from 'svelte';

	let { children, variant, weight }: { children: Snippet; variant?: string; weight?: number } =
		$props();

	const family = $derived(`"Llama Sans ${variant ?? view.variant}", ${fallback}`);

	function onkeydown(e: KeyboardEvent) {
		const t = e.target as HTMLElement;

		if (variant || e.metaKey || e.ctrlKey || e.altKey || e.key.toLowerCase() !== 'v') return;

		const typing =
			t.isContentEditable ||
			(/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) &&
				!['checkbox', 'radio', 'range'].includes((t as HTMLInputElement).type));

		if (typing) return;

		const { variants } = specimen;

		view.variant = variants[(variants.indexOf(view.variant) + 1) % variants.length];
	}
</script>

<svelte:window {onkeydown} />

<div style:--display={family} style:--w={weight ?? view.weight} class="specimen wrap">
	{@render children()}
</div>

<style>
	.wrap {
		display: grid;
		gap: 56px;
		max-width: 1180px;
		margin: 0 auto;
		padding: 40px max(16px, env(safe-area-inset-right, 0px)) 80px
			max(16px, env(safe-area-inset-left, 0px));
	}
	.wrap > :global(*) {
		margin: 0;
		min-width: 0;
	}
	/* a section label sits 12px above its section, and its note just under it */
	.wrap > :global(h2) {
		margin-bottom: -44px;
		font: 500 11px/1.4 var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.wrap > :global(h2 + p) {
		margin-bottom: -38px;
		font-size: 13px;
		color: var(--muted);
	}
	/* the design notes close on the next pass and the licence */
	.wrap > :global(.notes + p) {
		max-width: 70ch;
		margin-top: -42px;
		color: var(--muted);
	}
	/* room at the bottom on phones, where the controls float over the page; there the notes step aside */
	@media (max-width: 600px) {
		.wrap {
			padding-bottom: calc(150px + env(safe-area-inset-bottom, 0px));
		}
		.wrap > :global(:is(h2 + p, h2:has(+ .notes), .notes + p)) {
			display: none;
		}
	}
</style>
