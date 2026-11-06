<!--
	A large workstation: a tall tower with two handles on its top, its front a lattice of round vents, its side a
	plain panel, the power light in cinder. After the deskside AI towers (a DGX Station, a Mac Pro). A motif for posts
	about the biggest models on a desk.
-->
<script lang="ts">
	import './motif.css';
	import { type Box, device, pen, shadow } from './box';
	import Solid from './Solid.svelte';

	const view = device(96, 248);
	const body: Box = { max: [0.55, 2.0, 0.9], min: [-0.55, 0, -0.9] };
	const handles: Box[] = [
		{ max: [0.42, 2.18, -0.62], min: [-0.42, 2.0, -0.76] },
		{ max: [0.42, 2.18, 0.76], min: [-0.42, 2.0, 0.62] }
	];
	const d = pen(body, view);

	const lattice = Array.from({ length: 6 * 13 }, (_, i) => [
		0.17 + (i % 6) * 0.153,
		0.18 + Math.floor(i / 6) * 0.135
	]);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<polygon class="floor" points={shadow(body, view)} />

	<Solid box={handles[1]} {view} />

	<Solid box={body} {view} />

	<Solid box={handles[0]} {view} />

	{#each lattice as [u, v] (u + '-' + v)}
		<polygon style:opacity="0.75" class="detail" points={d.circle('front', u, v, 0.05, 14)} />
	{/each}

	<polygon style:opacity="0.35" class="detail" points={d.rect('right', 0.12, 0.12, 1.56, 1.76)} />

	<polygon class="accent" points={d.circle('front', 0.92, 1.92, 0.032, 12)} />
</svg>
