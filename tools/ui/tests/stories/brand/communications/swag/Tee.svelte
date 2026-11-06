<!--
	A flat t-shirt front drawn in centimetres, with one print fitted into its placement as the Swag Creator places it:
	the print area measured from the centre front of the neckline, the artwork contained in it. With `guides`, it draws
	the three placements' areas instead of a print.
-->
<script lang="ts">
	import { type Artwork, type Fabric, type Placement, PLACEMENTS } from './prints';
	import { quantized, QUANTIZED_SIZE } from './quantized';

	interface Props {
		artwork?: Artwork;
		fabric: Fabric;
		guides?: boolean;
		placement?: Placement;
	}

	let { artwork, fabric, guides = false, placement }: Props = $props();

	// the neckline's lowest point at the centre front, where placements are measured from
	const NECK = 2;
	const uid = $props.id();

	const area = (p: Placement, scale = { height: 1, width: 1 }) => {
		const width = p.width * scale.width;
		const height = p.height * scale.height;

		return { height, width, x: p.x - width / 2, y: NECK + p.top };
	};

	const box = $derived(placement && artwork ? area(placement, artwork.printScale) : undefined);
	const line = $derived(fabric.tone === 'light' ? 'rgb(0 0 0 / 0.14)' : 'rgb(255 255 255 / 0.16)');
</script>

<svg
	aria-label="{artwork?.label ?? 'Placements'} on a {fabric.label.toLowerCase()} t-shirt"
	class="tee"
	role="img"
	viewBox="-40 -5 80 80"
>
	<path
		d="M-9 -2 L-22 1 L-37 13 L-31 25 L-26 21 L-27 72 L27 72 L26 21 L31 25 L37 13 L22 1 L9 -2 Q0 6 -9 -2 Z"
		fill={fabric.swatch}
		stroke={line}
		stroke-linejoin="round"
		stroke-width="0.3"
	/>

	<path d="M-9 -2 Q0 6 9 -2" fill="none" stroke={line} stroke-width="0.6" />

	{#if guides}
		{#each PLACEMENTS as p (p.id)}
			{@const a = area(p)}
			<rect
				fill="none"
				height={a.height}
				stroke="var(--art-cinder)"
				stroke-dasharray="0.8 0.6"
				stroke-width="0.25"
				width={a.width}
				x={a.x}
				y={a.y}
			/>
		{/each}
	{:else if box && artwork}
		{#if artwork.src}
			<image
				height={box.height}
				href={artwork.src[fabric.tone]}
				width={box.width}
				x={box.x}
				y={box.y}
			/>
		{:else}
			<svg
				height={box.height}
				viewBox="0 0 {QUANTIZED_SIZE.width} {QUANTIZED_SIZE.height}"
				width={box.width}
				x={box.x}
				y={box.y}
			>
				{@html quantized(fabric.tone, `q${uid}`)}
			</svg>
		{/if}
	{/if}
</svg>

<style>
	.tee {
		display: block;
		width: 100%;
		height: auto;
	}
</style>
