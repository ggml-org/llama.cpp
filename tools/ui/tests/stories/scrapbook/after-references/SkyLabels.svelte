<!--
	D · Labels on the sky, after the references' glass pills: a quantised sky running to the edges, and what Llama
	stands for in three glass labels, stepped down the piece, each in its own column. The lockup sits at the bottom,
	centred, small. Words only where they carry weight: three of them, and the name.
-->
<script lang="ts">
	import { type FormatKey, FORMATS, LOCKUP } from '../../brand/communications/posts/post';
	import PostFrame from '../../brand/communications/posts/PostFrame.svelte';
	import QuantSky from './QuantSky.svelte';

	interface Props {
		format: FormatKey;
		/** each label and the column it starts at, of six */
		labels?: [string, number][];
		sky: 'day' | 'night';
	}

	let {
		format,
		labels = [
			['Private', 0],
			['Offline', 3],
			['Open source', 1]
		],
		sky
	}: Props = $props();
</script>

<PostFrame
	format={FORMATS[format]}
	label="Labels on the sky, {FORMATS[format].name}, {sky}"
	tone={sky}
>
	<div class="bleed">
		<QuantSky
			columns={8}
			rows={Math.round((FORMATS[format].height / FORMATS[format].width) * 8)}
			seed={5}
			{sky}
		/>
	</div>

	<div class="labels">
		{#each labels as [label, column] (label)}
			<span style:--col={column} class="label">{label}</span>
		{/each}
	</div>

	<p class="lockup">{LOCKUP}Llama</p>
</PostFrame>

<style>
	.bleed {
		position: absolute;
		inset: calc(var(--m) * -1 - var(--safe-top)) calc(var(--m) * -1)
			calc(var(--m) * -1 - var(--safe-bottom));
	}
	.labels {
		position: relative;
		display: grid;
		gap: calc(var(--u) * 1.6);
		justify-items: start;
		margin-block: auto;
	}
	/* the glass label: the ground's light at a fifth, a hairline edge, blur behind; no shadow and no glow */
	.label {
		margin-left: calc((100% + var(--gutter)) / 6 * var(--col));
		padding: calc(var(--u) * 0.7) calc(var(--u) * 1.4);
		border: 2px solid rgb(255 255 255 / 0.28);
		border-radius: 999px;
		background: rgb(17 26 42 / 0.35);
		backdrop-filter: blur(calc(var(--u) * 0.8));
		font: 500 calc(var(--u) * 1.3) / 1 var(--sans);
		letter-spacing: -0.01em;
	}
	:global(.ground-day) .label {
		border-color: rgb(255 255 255 / 0.7);
		background: rgb(255 255 255 / 0.45);
	}
	.lockup {
		position: relative;
		margin: 0;
		font: 700 calc(var(--u) * 0.8) / 1 var(--sans);
		text-align: center;
	}
</style>
