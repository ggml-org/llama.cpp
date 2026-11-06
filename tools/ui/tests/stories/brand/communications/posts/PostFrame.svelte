<!--
	A social post or link preview at its real pixel size, scaled down to fit its column. The tone sets its ground and
	inks: the neutrals, light and dark (where both lockups may go), or llama.app's hero blue and its navy (the Llama
	lockup only). Its parts (PostHeader, ModelCard, PostChip, PostFooter) take their colours from these variables.
-->
<script lang="ts">
	import '../../fonts/fonts.css';
	import type { PostTone } from './post';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		height: number;
		label: string;
		tone: PostTone;
		width: number;
	}

	let { children, height, label, tone, width }: Props = $props();

	let column = $state(0);
	const scale = $derived(column ? Math.min(1, column / width) : 0);
</script>

<div bind:clientWidth={column} style:height="{height * scale}px" class="fit">
	<article
		style:height="{height}px"
		style:transform="scale({scale})"
		style:width="{width}px"
		aria-label={label}
		class="post {tone}"
	>
		{@render children()}
	</article>
</div>

<style>
	.fit {
		width: 100%;
		overflow: hidden;
		border-radius: 12px;
		box-shadow: 0 0 0 1px var(--border);
	}
	.post {
		--sans: 'Llama Sans', var(--font-sans);
		--mono: 'Llama Mono', var(--font-mono);
		position: relative;
		display: flex;
		flex-direction: column;
		box-sizing: border-box;
		overflow: hidden;
		transform-origin: 0 0;
		background: var(--p-bg);
		color: var(--p-ink);
		font: 400 24px / 1.4 var(--sans);
		/* the cards' meta lines and repo ids keep their columns */
		font-variant-numeric: tabular-nums;
	}

	.light {
		--p-bg: #fafafa;
		--p-ink: #111111;
		--p-muted: rgb(17 17 17 / 0.62);
		--p-line: rgb(17 17 17 / 0.1);
		--p-card: #ffffff;
		--p-inset: var(--neutral-100);
		--p-solid: #111111;
		--p-solid-ink: #fafafa;
		--p-link: var(--art-blue);
	}
	.dark {
		--p-bg: #111111;
		--p-ink: #fafafa;
		--p-muted: rgb(250 250 250 / 0.62);
		--p-line: rgb(250 250 250 / 0.13);
		--p-card: rgb(250 250 250 / 0.04);
		--p-inset: rgb(250 250 250 / 0.06);
		--p-solid: #fafafa;
		--p-solid-ink: #111111;
		--p-link: var(--art-sky);
	}
	/* llama.app's hero and its navy night version, as in Brand/Logo/Usage */
	.hero {
		--p-bg:
			radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%),
			radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%),
			linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%);
		--p-ink: #111111;
		--p-muted: rgb(17 17 17 / 0.62);
		--p-line: rgb(17 17 17 / 0.1);
		--p-card: rgb(255 255 255 / 0.9);
		--p-inset: rgb(17 17 17 / 0.05);
		--p-solid: #111111;
		--p-solid-ink: #fafafa;
		--p-link: var(--art-blue);
	}
	.navy {
		--p-bg:
			radial-gradient(ellipse at 15% 10%, #22385c 0%, transparent 55%),
			radial-gradient(ellipse at 90% 95%, #1e3752 0%, transparent 50%),
			linear-gradient(160deg, #152238 0%, #172a42 55%, #1a3049 100%);
		--p-ink: #fafafa;
		--p-muted: rgb(250 250 250 / 0.66);
		--p-line: rgb(250 250 250 / 0.14);
		--p-card: rgb(17 26 42 / 0.82);
		--p-inset: rgb(250 250 250 / 0.07);
		--p-solid: #fafafa;
		--p-solid-ink: #111111;
		--p-link: var(--art-sky);
	}

	/* the type the templates share: the headline, its lede, and a Mono label over a section (at a low specificity,
	   so the templates set their own sizes and margins) */
	:global(:where(.post) .headline) {
		margin: 0;
		font-family: var(--sans);
		font-weight: 700;
		line-height: 0.98;
		letter-spacing: -0.02em;
		text-wrap: balance;
	}
	:global(:where(.post) .lede) {
		margin: 0;
		color: var(--p-muted);
		font-family: var(--sans);
		line-height: 1.35;
		text-wrap: pretty;
	}
	:global(:where(.post) .label) {
		margin: 0;
		color: var(--p-muted);
		font: 500 22px / 1 var(--mono);
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}
</style>
