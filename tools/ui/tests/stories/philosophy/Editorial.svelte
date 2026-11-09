<!--
	The page the design philosophy is set on: an editorial column of running text at a reading size, with figures that
	break out wider and pull quotes in Display. Numbered section heads, the numbers in cinder (the mark on a detail),
	and the chapter navigation fixed at the right.
-->
<script lang="ts">
	import '../brand/fonts/fonts.css';
	import Chapters from './Chapters.svelte';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();

	let article: HTMLElement | undefined = $state();
</script>

<article bind:this={article} class="editorial">{@render children()}</article>

<Chapters root={article} />

<style>
	.editorial {
		--mono: 'Llama Mono', ui-monospace, 'SF Mono', Menlo, monospace;
		--text: min(40rem, calc(100% - 32px));
		display: grid;
		grid-template-columns:
			[full-start] minmax(16px, 1fr) [wide-start] minmax(0, 9rem) [text-start] var(--text)
			[text-end] minmax(0, 9rem) [wide-end] minmax(16px, 1fr) [full-end];
		counter-reset: section;
		min-height: 100vh;
		padding: clamp(40px, 8vw, 112px) 0 clamp(64px, 10vw, 160px);
		background: var(--surface-page);
		color: var(--fg-default);
		font: 400 18px / 1.6 var(--font-sans);
		text-wrap: pretty;
	}

	.editorial > :global(*) {
		grid-column: text;
		margin: 0;
		min-width: 0;
	}

	.editorial > :global(* + *) {
		margin-top: 1.1em;
	}

	.editorial > :global(.wide) {
		grid-column: wide;
	}

	/* the masthead: a kicker, the title in Display, the lead and a rule under it */
	.editorial > :global(.kicker) {
		color: var(--fg-muted);
		font: 500 12px / 1.4 var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.editorial > :global(h1) {
		grid-column: wide;
		margin-top: 20px;
		font-size: clamp(48px, 9vw, 112px);
		font-weight: 700;
		letter-spacing: -0.01em;
		line-height: 1;
	}

	.editorial > :global(.lead) {
		margin-top: 32px;
		color: var(--fg-muted);
		font-size: clamp(20px, 2.4vw, 24px);
		line-height: 1.45;
		text-wrap: balance;
	}

	.editorial > :global(.lead strong) {
		color: var(--fg-default);
		font-weight: 500;
	}

	/* each section opens on a hairline, its number above the heading */
	.editorial > :global(h2) {
		counter-increment: section;
		margin-top: clamp(72px, 10vw, 120px);
		padding-top: 20px;
		border-top: 1px solid var(--line-default);
		font-size: clamp(28px, 3.6vw, 40px);
		font-weight: 600;
		letter-spacing: -0.005em;
		line-height: 1.15;
		text-wrap: balance;
	}

	.editorial > :global(h2)::before {
		content: counter(section, decimal-leading-zero);
		display: block;
		margin-bottom: 16px;
		color: var(--art-cinder);
		font: 500 13px / 1 var(--mono);
		letter-spacing: 0.04em;
	}

	.editorial > :global(h2 + *) {
		margin-top: 24px;
	}

	.editorial :global(strong) {
		font-weight: 600;
	}

	.editorial :global(em) {
		font-style: italic;
	}

	.editorial :global(a) {
		text-decoration: underline;
		text-decoration-color: var(--line-default);
		text-underline-offset: 3px;
	}

	.editorial :global(:not(pre) > code) {
		padding: 0.1em 0.3em;
		border-radius: 6px;
		background: var(--surface-code);
		font: 0.85em var(--mono);
	}

	/* a footnote marker, and the note it points to: small and muted, under the figure */
	.editorial :global(sup) {
		color: var(--art-cinder);
		font: 500 0.75em var(--mono);
		line-height: 0;
	}

	.editorial > :global(.note) {
		color: var(--fg-muted);
		font-size: 14px;
		line-height: 1.5;
	}

	.editorial > :global(ul) {
		padding-left: 1.2em;
		list-style: disc;
	}

	.editorial > :global(ul li + li) {
		margin-top: 0.5em;
	}

	.editorial > :global(ul li::marker) {
		color: var(--fg-muted);
	}

	/* a pull quote: Display size, wider than the text, with a cinder rule */
	.editorial > :global(.pull) {
		grid-column: wide;
		margin-block: clamp(40px, 6vw, 72px);
		padding-left: clamp(16px, 3vw, 32px);
		border-left: 2px solid var(--art-cinder);
		font-size: clamp(28px, 4.4vw, 48px);
		font-weight: 600;
		letter-spacing: -0.01em;
		line-height: 1.12;
		text-wrap: balance;
	}

	.editorial > :global(.pull + *) {
		margin-top: 0;
	}

	/* the closing rules: big numbers in the margin */
	.editorial > :global(ol) {
		counter-reset: rule;
		display: grid;
		gap: 20px;
		margin-top: 32px;
		list-style: none;
	}

	.editorial > :global(ol li) {
		counter-increment: rule;
		display: grid;
		grid-template-columns: 3rem 1fr;
		align-items: baseline;
		padding-bottom: 20px;
		border-bottom: 1px solid var(--line-default);
		font-size: 20px;
		line-height: 1.45;
	}

	.editorial > :global(ol li)::before {
		content: counter(rule);
		color: var(--fg-muted);
		font: 500 13px var(--mono);
	}

	.editorial > :global(.colophon) {
		margin-top: 72px;
		color: var(--fg-muted);
		font: 400 13px / 1.6 var(--mono);
	}
</style>
