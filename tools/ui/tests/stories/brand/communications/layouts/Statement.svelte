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
		/** the command's `$`; off for a line of config */
		prompt?: boolean;
		headline: string;
		lede?: string;
		motif?: Snippet;
	}

	let { command, footer, format, header, headline, lede, motif, overlay, prompt, tone }: Props =
		$props();
</script>

<PostFrame
	format={FORMATS[format]}
	label="Statement, {FORMATS[format].name}, {tone}"
	{overlay}
	{tone}
>
	<PostHeader {...header} />

	<div class:plain={!motif} class="body">
		{#if motif}<div class="motif fill">{@render motif()}</div>{/if}

		<div class="text">
			<h2 class="t-display">{headline}</h2>

			{#if lede}<p class="t-lede">{lede}</p>{/if}

			{#if command}<Command {command} {prompt} />{/if}
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
	.text :global(.terminal) {
		margin-top: calc(var(--stack-after) - var(--stack));
	}

	:global(.shape-wide) .body {
		grid-template-rows: none;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		column-gap: var(--gutter);
	}
	:global(.shape-wide) .motif {
		grid-area: 1 / 8 / 2 / -1;
	}
	:global(.shape-wide) .text {
		grid-area: 1 / 1 / 2 / 8;
	}
	:global(.shape-wide) .plain .text {
		grid-column: 1 / -1;
	}
</style>
