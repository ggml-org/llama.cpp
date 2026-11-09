<!--
	C · Split field, after the references' split posters: the top of the piece a field of artwork running to the
	edges (a quantised sky), the bottom a plain panel with the headline, a muted paragraph at a large size, and a
	small footnote with the icon at the bottom. The header floats over the field in a glass bar.
-->
<script lang="ts">
	import {
		type FormatKey,
		FORMATS,
		type Ground,
		ICON,
		LOCKUP
	} from '../../brand/communications/posts/post';
	import PostFrame from '../../brand/communications/posts/PostFrame.svelte';
	import QuantSky from './QuantSky.svelte';

	interface Props {
		footnote?: string;
		format: FormatKey;
		headline?: string;
		paragraph?: string;
		/** the field's share of the piece's height */
		split?: number;
		tone: Ground;
	}

	let {
		footnote = 'Free and open source, by the llama.cpp team and Hugging Face.',
		format,
		headline = 'A model in 4 bits, on the laptop you already have.',
		paragraph = 'Quantised weights take a model down to about a quarter of its size, so it fits in the memory your laptop has.',
		split = 0.42,
		tone
	}: Props = $props();

	const sky = $derived(tone === 'dark' || tone === 'night' ? 'night' : 'day');
	const rows = $derived(
		Math.round((FORMATS[format].height * split) / (FORMATS[format].width / 12))
	);
</script>

<PostFrame format={FORMATS[format]} label="Split field, {FORMATS[format].name}, {tone}" {tone}>
	<div style:--field="{FORMATS[format].height * split}px" class="field">
		<QuantSky columns={12} {rows} {sky} />
	</div>

	<div style:--field="{FORMATS[format].height * split}px" class="bar">
		<span class="lockup">{LOCKUP}Llama</span>

		<span class="url">llama.app</span>
	</div>

	<div style:--field="{FORMATS[format].height * split}px" class="panel">
		<h2 class="headline">{headline}</h2>

		<p class="paragraph">{paragraph}</p>

		<footer class="foot">
			<span>{footnote}</span>

			<span aria-hidden="true" class="icon">{ICON}</span>
		</footer>
	</div>
</PostFrame>

<style>
	.field {
		position: absolute;
		inset: calc(var(--m) * -1 - var(--safe-top)) calc(var(--m) * -1) auto;
		height: var(--field);
	}
	/* the glass bar: the ground's card over the field, a hairline, no shadow */
	.bar {
		position: absolute;
		inset: 0 0 auto;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: calc(var(--u) * 2);
		padding: calc(var(--u) * 0.5) calc(var(--u) * 0.7);
		border: 2px solid rgb(255 255 255 / 0.35);
		border-radius: 999px;
		background: rgb(255 255 255 / 0.22);
		backdrop-filter: blur(calc(var(--u) * 0.6));
		color: #111111;
	}
	:global(.ground-dark) .bar,
	:global(.ground-night) .bar {
		border-color: rgb(255 255 255 / 0.16);
		background: rgb(255 255 255 / 0.08);
		color: #fafafa;
	}
	.lockup {
		font: 700 calc(var(--u) * 0.72) / 1 var(--sans);
	}
	.url {
		font: 400 calc(var(--u) * 0.58) / 1 var(--mono);
	}
	/* the panel starts where the field ends, 1.6u under it */
	.panel {
		display: flex;
		flex: 1;
		flex-direction: column;
		margin-top: calc(var(--field) - var(--m) - var(--safe-top) + var(--u) * 1.6);
	}
	.headline {
		max-width: calc(var(--u) * 19);
		margin: 0;
		font: 600 calc(var(--u) * 1.5) / 1.06 var(--sans);
		letter-spacing: -0.03em;
		text-wrap: balance;
	}
	.paragraph {
		max-width: calc(var(--u) * 18);
		margin: calc(var(--u) * 1.2) 0 0;
		color: var(--p-muted);
		font: 500 calc(var(--u) * 0.96) / 1.24 var(--sans);
		letter-spacing: -0.02em;
		text-wrap: pretty;
	}
	.foot {
		display: flex;
		align-items: end;
		justify-content: space-between;
		gap: calc(var(--u) * 2);
		margin-top: auto;
		color: var(--p-muted);
		font-size: calc(var(--u) * 0.58);
		line-height: 1.3;
	}
	.icon {
		color: var(--p-ink);
		font: 700 calc(var(--u) * 1.2) / 1 var(--sans);
	}
</style>
