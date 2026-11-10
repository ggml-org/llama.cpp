<!--
	"Light enough to forget it's there": three facts, each a card with its drawing pinned to the bottom.
	- sizes: the download and the app, on one scale with a photo for comparison, each a bar in Mono figures
	- storage: Llama, llama.cpp and other tools wired to the one model file in the Hugging Face cache
	- idle: memory over time, nothing, then in use, five minutes idle, and unloaded, as a step chart with Mono phases
	With `arch`, Llama's own things are in its sky, the tools are apps (cinder) and the model file is grey.
-->
<script lang="ts">
	import { Image, Package } from '@lucide/svelte';
	import qwen from '$lib/assets/model-logos/qwen.svg?raw';
	import { Logo } from '$lib/components/app/misc';
	import { Badge } from '$lib/components/ui/badge';
	import type { Snippet } from 'svelte';

	let { arch = false }: { arch?: boolean } = $props();

	// on a 4 MB scale: the app, the download, and a phone photo for the "smaller than a photo" claim
	const sizes = [
		{ icon: 'dmg', mb: 1, name: 'Llama.dmg', note: 'Download' },
		{ icon: 'app', mb: 4, name: 'Llama.app', note: 'Installed' },
		{ icon: 'photo', mb: 3, name: 'A phone photo', note: 'For scale' }
	];
	const tools = ['Llama', 'llama.cpp', 'Other tools'];
</script>

{#snippet card(title: string, body: string, art: Snippet)}
	<section class="panel card">
		<div>
			<h4>{title}</h4>
			<p class="muted body">{body}</p>
		</div>
		<div class="art" aria-hidden="true">{@render art()}</div>
	</section>
{/snippet}

{#snippet sizeArt()}
	<ul class="sizes">
		{#each sizes as s (s.name)}
			<li class={{ photo: s.icon === 'photo' }}>
				<span class="row">
					<span class={['tile', { lib: s.icon !== 'photo' }]}>
						{#if s.icon === 'app'}<Logo style="--size: 13px" />{:else if s.icon === 'dmg'}<Package
								size={14}
							/>{:else}<Image size={14} />{/if}
					</span>
					<b>{s.name}</b>
					<span class="kicker">{s.note}</span>
					<code>{s.mb} MB</code>
				</span>
				<span class="bar"><i style:width="{(s.mb / 4) * 100}%"></i></span>
			</li>
		{/each}
	</ul>
{/snippet}

{#snippet storageArt()}
	<div class="storage">
		<div class="tools">
			{#each tools as t (t)}<span class={['part', t === 'Llama' ? 'lib' : 'app']}>{t}</span>{/each}
		</div>
		<!-- three wires that meet over the file -->
		<svg class="joins" fill="none" preserveAspectRatio="none" viewBox="0 0 300 28">
			<path d="M50 0V8Q50 14 56 14H144Q150 14 150 20V28M250 0V8Q250 14 244 14H156Q150 14 150 20M150 0V28" />
		</svg>
		<span class="part be file">
			<span class="mark">{@html qwen}</span>
			<span class="text"
				><b>Qwen3.8 <Badge variant="outline">27B</Badge></b><span class="kicker">In the Hugging Face cache</span
				></span
			>
			<code>19 GB</code>
		</span>
	</div>
{/snippet}

{#snippet idleArt()}
	<div class="idle">
		<svg fill="none" preserveAspectRatio="none" viewBox="0 0 200 100">
			<path class="use" d="M20 100V10H80V100Z" />
			<path class="wait" d="M80 100V10H140V100Z" />
			<path class="line" d="M0 99.5H20V10H140V99.5H200" vector-effect="non-scaling-stroke" />
		</svg>
		<span class="phase" style:left="10%">In use</span>
		<span class="phase" style:left="40%">5 min idle</span>
		<span class="phase low" style:left="70%">Unloaded</span>
		<span class="axis y">Memory</span>
		<span class="axis x">Time →</span>
	</div>
{/snippet}

<div class={['lf', { arch }]}>
	<div class="cards">
		{@render card('4 MB app', 'A native Mac app, and just a 1 MB download — smaller than a photo.', sizeArt)}
		{@render card(
			'Each model stored once',
			'Kept in the Hugging Face cache, shared with llama.cpp and other tools.',
			storageArt
		)}
		{@render card(
			'Nothing loaded when idle',
			'Models load when something asks for one and unload after 5 minutes idle.',
			idleArt
		)}
	</div>
</div>

<style>
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr));
		gap: 12px;
	}
	.card {
		align-content: space-between;
		gap: 28px;
	}
	h4 {
		margin: 0;
		font-size: 15px;
		font-weight: 600;
	}
	.body {
		margin: 4px 0 0;
		font-size: 14px;
	}
	.art {
		font-size: 12px;
	}

	/* sizes: a row each, a bar under it on the 4 MB scale */
	.sizes {
		display: grid;
		gap: 12px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.row b {
		font-weight: 500;
	}
	.row .kicker {
		flex: 1;
		font-size: 10px;
	}
	.tile {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border: 1px solid var(--line-default);
		border-radius: 6px;
		background: var(--surface-raised);
		color: var(--figure-muted);
	}
	.tile.lib {
		border-color: var(--lib-line);
		background: var(--lib-bg);
		color: var(--lib);
	}
	.bar {
		display: block;
		height: 4px;
		margin: 6px 0 0 32px;
		border-radius: 2px;
		background: color-mix(in oklab, var(--fg-default) 6%, transparent);
	}
	.bar i {
		display: block;
		height: 100%;
		border-radius: 2px;
		background: var(--lib);
	}
	.lf:not(.arch) .bar i {
		background: var(--fg-default);
	}
	.photo {
		color: var(--figure-muted);
	}
	.photo .bar i {
		background: var(--wire) !important;
	}

	/* storage: the tools, the wires, the file */
	.storage {
		display: grid;
	}
	.tools {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 6px;
	}
	.lf .tools .part {
		justify-content: center;
		padding: 5px 4px;
		font-size: 12px;
		text-align: center;
	}
	.tools .part.app {
		color: var(--app);
	}
	.tools .part.lib {
		color: var(--lib);
	}
	.joins {
		width: 100%;
		height: 28px;
		stroke: var(--wire);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.joins path {
		vector-effect: non-scaling-stroke;
	}
	.lf .file {
		display: flex;
		gap: 10px;
		align-items: center;
		padding: 8px 10px;
	}
	.file .text {
		display: grid;
		flex: 1;
		gap: 2px;
	}
	.file b {
		display: flex;
		gap: 6px;
		align-items: center;
		font-weight: 500;
	}
	.file .kicker {
		font-size: 10px;
	}

	/* idle: a step chart, the phases set in Mono */
	.idle {
		position: relative;
		height: 112px;
		margin: 0 0 0 14px;
		border-left: 1px solid var(--line-default);
	}
	.idle svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
	}
	.use {
		fill: var(--lib);
		fill-opacity: 0.2;
	}
	.wait {
		fill: var(--lib);
		fill-opacity: 0.07;
	}
	.lf:not(.arch) .use,
	.lf:not(.arch) .wait {
		fill: var(--fg-default);
	}
	.lf:not(.arch) .use {
		fill-opacity: 0.1;
	}
	.lf:not(.arch) .wait {
		fill-opacity: 0.04;
	}
	.line {
		stroke: var(--lib);
		stroke-width: 1.5;
	}
	.phase {
		position: absolute;
		top: 50%;
		width: 30%;
		color: var(--fg-default);
		font: 500 10px / 1 var(--font-mono);
		letter-spacing: 0.06em;
		text-align: center;
		text-transform: uppercase;
	}
	.phase.low {
		top: auto;
		bottom: 8px;
		color: var(--figure-muted);
	}
	.axis {
		position: absolute;
		color: var(--figure-muted);
		font: 500 9px / 1 var(--font-mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
	.axis.y {
		top: 0;
		left: -14px;
		transform: rotate(-90deg) translateX(-100%);
		transform-origin: top left;
	}
	.axis.x {
		right: 0;
		bottom: -16px;
	}
</style>
