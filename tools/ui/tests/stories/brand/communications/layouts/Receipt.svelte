<!--
	Receipt: a headline in Llama Sans, and a receipt beside or under it in Llama Mono, its rows the facts as data and
	its total in cinder. The receipt makes the point the headline does not have to: what it cost, what it took, what
	it asked for. On a wide format it sits to the right; otherwise under the headline.
-->
<script lang="ts">
	import { type FormatKey, FORMATS, type Ground } from '../posts/post';
	import PostFooter from '../posts/PostFooter.svelte';
	import PostFrame from '../posts/PostFrame.svelte';
	import PostHeader from '../posts/PostHeader.svelte';
	import type { Header } from './layout';

	interface Props {
		footer?: [string, string];
		format: FormatKey;
		header: Header;
		headline: string;
		overlay?: boolean;
		/** the receipt's lines, the last one the total, in cinder */
		rows: [string, string][];
		/** the line over the receipt */
		title: string;
		tone: Ground;
	}

	let { footer, format, header, headline, overlay, rows, title, tone }: Props = $props();
</script>

<PostFrame
	format={FORMATS[format]}
	label="Receipt, {FORMATS[format].name}, {tone}"
	{overlay}
	{tone}
>
	<PostHeader {...header} />

	<div class="body">
		<h2 class="t-display">{headline}</h2>

		<dl class="receipt">
			<dt class="title">{title}</dt>

			{#each rows as [label, value], i (label)}
				<div class:total={i === rows.length - 1}>
					<dt>{label}</dt>

					<dd>{value}</dd>
				</div>
			{/each}
		</dl>
	</div>

	{#if footer}<PostFooter left={footer[0]} right={footer[1]} />{/if}
</PostFrame>

<style>
	.body {
		display: grid;
		flex: 1;
		grid-template-rows: auto 1fr;
		gap: var(--u);
		margin-block: var(--u);
	}
	.receipt {
		display: grid;
		align-self: end;
		gap: calc(var(--u) * 0.45);
		margin: 0;
		padding: calc(var(--u) * 0.9) calc(var(--u) * 1);
		border: 2px solid var(--p-line);
		border-radius: calc(var(--u) * 0.5);
		background: var(--p-card);
		font: 400 calc(var(--u) * 0.62) / 1.2 var(--mono);
	}
	.title {
		margin-bottom: calc(var(--u) * 0.25);
		color: var(--p-muted);
		font: 500 calc(var(--u) * 0.48) / 1 var(--mono);
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}
	.receipt div {
		display: flex;
		justify-content: space-between;
		gap: var(--u);
	}
	dt {
		color: var(--p-muted);
	}
	dd {
		margin: 0;
	}
	/* the total: under a dashed rule, larger, its figure the one accent */
	.total {
		margin-top: calc(var(--u) * 0.3);
		padding-top: calc(var(--u) * 0.6);
		border-top: 2px dashed var(--p-line);
		font-size: calc(var(--u) * 0.9);
	}
	.total dt {
		color: var(--p-ink);
	}
	.total dd {
		color: var(--art-cinder);
		font-weight: 600;
	}

	/* the headline steps down where there is less height to share: a little on tall formats, more on square ones */
	:global(.shape-tall) .t-display {
		font-size: calc(var(--u) * 2.8);
	}
	:global(.shape-square) .t-display {
		font-size: calc(var(--u) * 2.2);
	}

	:global(.shape-wide) .body {
		grid-template-rows: none;
		grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
		column-gap: var(--gutter);
	}
	:global(.shape-wide) .t-display {
		grid-column: 1 / 8;
		font-size: calc(var(--u) * 2.4);
		align-self: end;
	}
	:global(.shape-wide) .receipt {
		grid-column: 8 / -1;
	}
</style>
