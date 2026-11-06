<!--
	Playing with the mark: the quantized L from Communications › Swag, large on a light plate with its bit-depth legend,
	and printed on a black t-shirt as the Swag Creator places it
-->
<script lang="ts">
	import { ARTWORKS, FABRICS, PLACEMENTS } from '../../brand/communications/swag/prints';
	import { quantized, QUANTIZED_SIZE } from '../../brand/communications/swag/quantized';
	import Tee from '../../brand/communications/swag/Tee.svelte';

	const artwork = ARTWORKS.find((a) => a.id === 'quantized')!;
	const black = FABRICS.find((f) => f.id === 'black')!;
	const center = PLACEMENTS.find((p) => p.id === 'center')!;
</script>

<div class="quantized">
	<div class="plate rounded-card elevated-surface">
		<svg
			aria-label="The quantized L, from 16-bit down to 2-bit"
			role="img"
			viewBox="0 0 {QUANTIZED_SIZE.width} {QUANTIZED_SIZE.height}"
		>
			{@html quantized('light', 'philosophy-quantized')}
		</svg>
	</div>

	<div class="tee rounded-card elevated-surface">
		<Tee {artwork} fabric={black} placement={center} />
	</div>
</div>

<style>
	.quantized {
		display: grid;
		grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
		gap: 16px;
	}

	.plate,
	.tee {
		display: grid;
		place-items: center;
		padding: clamp(20px, 4vw, 40px);
	}

	.plate {
		background: #fafafa;
		color: #111111;
	}

	/* a fixed light ground, so the black shirt reads in both themes */
	.tee {
		background: #e9e9e7;
	}

	svg {
		width: 100%;
		height: auto;
	}

	@media (width < 640px) {
		.quantized {
			grid-template-columns: 1fr;
		}
	}
</style>
