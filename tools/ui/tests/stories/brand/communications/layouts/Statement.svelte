<!--
	Statement: one thing said big. The headline at display size, a lede and a command under it, and a motif filling
	the rest: beside the text on a wide format, above it on a square or tall one.
-->
<script lang="ts">
	import Command from '../posts/Command.svelte';
	import { FORMATS } from '../posts/post';
	import PostFooter from '../posts/PostFooter.svelte';
	import PostFrame from '../posts/PostFrame.svelte';
	import PostHeader from '../posts/PostHeader.svelte';
	import type { LayoutProps } from './layout';
	import type { Snippet } from 'svelte';

	interface Props extends LayoutProps {
		command?: string;
		/** the command's type size in units, larger for a short line */
		commandSize?: number;
		/** the command's `$`; off for a line of config */
		prompt?: boolean;
		/** on a wide format, the column the motif starts at (the text takes the ones before it) */
		split?: number;
		headline: string;
		lede?: string;
		/** the lede's type size in units, larger than 0.84 for a phone, tracked and led tighter to fit */
		ledeSize?: number;
		motif?: Snippet;
	}

	let {
		command,
		commandSize,
		footer,
		format,
		header,
		headline,
		lede,
		ledeSize,
		motif,
		overlay,
		prompt,
		split = 8,
		tone
	}: Props = $props();
</script>

<PostFrame
	format={FORMATS[format]}
	label="Statement, {FORMATS[format].name}, {tone}"
	{overlay}
	{tone}
>
	<PostHeader {...header} />

	<div class:plain={!motif} style:--split={split} class="body">
		{#if motif}<div class="motif fill">{@render motif()}</div>{/if}

		<div class="text">
			<h2 class="t-display">{headline}</h2>

			{#if lede}<p style:--lede={ledeSize} class={['t-lede', { large: ledeSize }]}>{lede}</p>{/if}

			{#if command}<Command {command} {prompt} size={commandSize} />{/if}
		</div>
	</div>

	{#if footer}<PostFooter left={footer[0]} right={footer[1]} />{/if}
</PostFrame>

<style>
	.body {
		display: grid;
		flex: 1;
		grid-template-rows: minmax(0, 1fr) auto;
		gap: calc(var(--u) * 1.5);
		min-height: 0;
		margin-block: calc(var(--u) * 1.5);
	}
	.text {
		display: grid;
		gap: var(--stack);
		justify-items: start;
		align-self: end;
	}
	/* a larger lede tracks and leads a little tighter, so it keeps its measure and its lines, balanced */
	.t-lede.large {
		font-size: calc(var(--u) * var(--lede));
		line-height: 1.25;
		letter-spacing: -0.015em;
		text-wrap: balance;
	}
	.text :global(.terminal) {
		margin-top: calc(var(--stack-after) - var(--stack));
	}

	/* a wide format is short: a unit over the text and a little less under it */
	:global(.shape-wide) .body {
		margin-block: calc(var(--u) * 1) calc(var(--u) * 0.8);
		grid-template-rows: none;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		column-gap: var(--gutter);
	}
	:global(.shape-wide) .motif {
		grid-area: 1 / var(--split) / 2 / -1;
	}
	:global(.shape-wide) .text {
		grid-area: 1 / 1 / 2 / var(--split);
	}
	:global(.shape-wide) .plain .text {
		grid-column: 1 / -1;
	}
</style>
