<!--
	A small workstation: a compact desktop box, low and square, its front a fine perforated mesh, its side with a few
	ports and the power button, the status light in cinder. After the small AI desktops (a DGX Spark, a Mac Studio).
	A motif for posts about bigger models on a desk.
-->
<script lang="ts">
	import './motif.css';
	import { type Box, device, front, pen, shadow } from './box';
	import Solid from './Solid.svelte';

	let { flat = false }: { flat?: boolean } = $props();

	// the isometric view, or straight at the front for the flat version
	const view = $derived(flat ? front(180, 206) : device(118, 175));
	const body: Box = { max: [0.85, 0.62, 0.85], min: [-0.85, 0, -0.85] };
	const d = $derived.by(() => pen(body, view));

	const mesh = Array.from({ length: 22 * 7 }, (_, i) => [
		0.12 + (i % 22) * 0.067,
		0.1 + Math.floor(i / 22) * 0.067
	]);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	{#if !flat}
		<polygon class="floor" points={shadow(body, view)} />{/if}

	<Solid box={body} {flat} {view} />

	{#if !flat}<polygon
			style:opacity="0.4"
			class="detail"
			points={d.rect('top', 0.08, 0.08, 1.54, 1.54)}
		/>{/if}

	{#each mesh as [u, v] (u + '-' + v)}
		<polygon class="hole" points={d.circle('front', u, v, 0.018, 8)} />
	{/each}

	{#each [0.2, 0.42] as u (u)}
		{#if !flat}<polygon class="detail" points={d.rect('right', u, 0.24, 0.14, 0.07)} />{/if}
	{/each}

	{#if !flat}<polygon class="detail" points={d.circle('right', 0.78, 0.27, 0.045)} />{/if}

	{#if !flat}<polygon class="detail" points={d.rect('right', 1.0, 0.22, 0.24, 0.1)} />{/if}

	{#if !flat}<polygon class="detail" points={d.circle('right', 1.45, 0.4, 0.06)} />{/if}

	<polygon class="accent" points={d.circle('front', 1.56, 0.56, 0.028, 12)} />
</svg>
