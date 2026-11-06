<!--
	The Microsoft Store's artwork. The app tile is the icon alone; the poster art, the Store's main logo, is the icon
	over the name; the super hero art is a picture with no words (the Store sets the name over it): the machines Llama
	runs on, a laptop, a small workstation and a server, side by side above the bottom quarter the Store may cover.
-->
<script lang="ts">
	import Laptop from '../posts/motifs/Laptop.svelte';
	import Server from '../posts/motifs/Server.svelte';
	import WorkstationSmall from '../posts/motifs/WorkstationSmall.svelte';
	import { FORMATS, type Ground, ICON } from '../posts/post';
	import PostFrame from '../posts/PostFrame.svelte';

	interface Props {
		kind: 'hero' | 'poster' | 'tile';
		overlay?: boolean;
		tone: Ground;
	}

	let { kind, overlay, tone }: Props = $props();

	const format = $derived(
		FORMATS[kind === 'hero' ? 'msHero' : kind === 'poster' ? 'msPoster' : 'msTile']
	);
</script>

<PostFrame {format} label="{format.name}, {tone}" {overlay} {tone}>
	{#if kind === 'hero'}
		<div class="machines">
			<div class="fill"><Laptop /></div>

			<div class="fill"><WorkstationSmall /></div>

			<div class="fill"><Server /></div>
		</div>
	{:else}
		<div class={['logo', kind]}>
			<span aria-label="Llama" class="icon" role="img">{ICON}</span>

			{#if kind === 'poster'}<span class="name">Llama</span>{/if}
		</div>
	{/if}
</PostFrame>

<style>
	.machines {
		display: grid;
		flex: 1;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: var(--gutter);
		min-height: 0;
	}
	.logo {
		display: flex;
		flex: 1;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: calc(var(--u) * 1.2);
		color: var(--p-ink);
	}
	/* the glyph's advance carries a side-bearing on the right: nudged so its ink sits in the middle */
	.icon {
		font: 700 calc(var(--u) * 14) / 1 var(--sans);
		transform: translateX(0.04em);
	}
	/* on the tile the icon fills the area; on the poster it stands over the name, a little above the middle */
	.tile .icon {
		font-size: calc(var(--u) * 16);
	}
	.poster {
		padding-bottom: calc(var(--u) * 4);
	}
	.name {
		font: 700 calc(var(--u) * 3.2) / 0.92 var(--sans);
		letter-spacing: -0.035em;
	}
</style>
