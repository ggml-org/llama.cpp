<!--
	Swag: the merchandise prints the Swag Creator (ggml-org/swag-creator) previews on 3D garments, shown flat here.
	First the quantized L, the one print drawn in code, on light and dark; then every print on each fabric at its
	placement, and the three placements measured as a print shop measures them.
-->
<script lang="ts">
	import '../../fonts/fonts.css';
	import { ARTWORKS, FABRICS, PLACEMENTS } from './prints';
	import { QUANTIZATION_LAYERS, quantized, QUANTIZED_SIZE } from './quantized';
	import Tee from './Tee.svelte';

	// where each print is shown on the sheet: the small marks at their small placements, the rest at the centre
	const SHEET = [
		{ artwork: 'icon', placement: 'center' },
		{ artwork: 'llama-logo', placement: 'left-chest' },
		{ artwork: 'llama-cpp-logo', placement: 'center' },
		{ artwork: 'quantized', placement: 'center' }
	].map(({ artwork, placement }) => ({
		artwork: ARTWORKS.find((a) => a.id === artwork)!,
		placement: PLACEMENTS.find((p) => p.id === placement)!
	}));

	// a print's side in centimetres, after the artwork's own scale
	const size = (side: number, scale = 1) => Math.round(side * scale * 10) / 10;

	const precision = (splits: 'smooth' | number) =>
		splits === 'smooth'
			? 'the true outline'
			: `${splits * splits} ${splits === 1 ? 'square' : 'squares'} a pixel`;
</script>

<div class="swag">
	<header class="intro">
		<h1>Swag</h1>

		<p>
			Prints for t-shirts and hoodies, as the Swag Creator previews them on 3D garments. Every print
			comes from llama-brand except the quantized L, which is drawn in code, so it stays exact at
			any size.
		</p>
	</header>

	<section class="feature">
		<div class="copy">
			<h2>L · Quantized</h2>

			<p>
				The Llama icon as a model is quantized: the true outline at the top, and half the precision
				at each step down, on llama.cpp's everyday ladder. It is printed only at the centre, the one
				size its detail reads at.
			</p>

			<ol class="ladder">
				{#each QUANTIZATION_LAYERS as layer (layer.label)}
					<li>
						<span class="bits">{layer.label}</span>

						<span>{precision(layer.splits)}, {layer.rows} rows</span>
					</li>
				{/each}
			</ol>
		</div>

		<div class="plates">
			{#each ['light', 'dark'] as const as tone (tone)}
				<figure class="plate rounded-card elevated-surface {tone}">
					<svg
						aria-label="L · Quantized, {tone}"
						role="img"
						viewBox="0 0 {QUANTIZED_SIZE.width} {QUANTIZED_SIZE.height}"
					>
						{@html quantized(tone, `feature-${tone}`)}
					</svg>

					<figcaption>
						{tone === 'light' ? 'Dark ink, for light fabric' : 'Light ink, for dark fabric'}
					</figcaption>
				</figure>
			{/each}
		</div>
	</section>

	<section class="sheet">
		<h2>Print sheet</h2>

		<p>
			Each print on the three fabrics, at its placement. The ink follows the fabric: dark on white
			and melange, light on black.
		</p>

		<div class="rows">
			{#each SHEET as item (item.artwork.id)}
				<div class="row">
					<div class="label">
						<span class="name">{item.artwork.label}</span>

						<span class="note">
							{item.placement.label} · {size(item.placement.width, item.artwork.printScale?.width)} ×
							{size(item.placement.height, item.artwork.printScale?.height)} cm
						</span>
					</div>

					{#each FABRICS as fabric (fabric.id)}
						<div class="tee">
							<Tee artwork={item.artwork} {fabric} placement={item.placement} />

							<span class="note">{fabric.label}</span>
						</div>
					{/each}
				</div>
			{/each}
		</div>
	</section>

	<section class="placements">
		<h2>Placements</h2>

		<div class="guide">
			<div class="tee"><Tee fabric={FABRICS[0]} guides /></div>

			<dl>
				{#each PLACEMENTS as p (p.id)}
					<div>
						<dt>{p.label}</dt>

						<dd>
							{p.width} × {p.height} cm, {p.top} cm below the neckline{p.x
								? `, ${p.x} cm to the wearer's left`
								: ', centred'}
						</dd>
					</div>
				{/each}
			</dl>
		</div>
	</section>
</div>

<style>
	.swag {
		--mono: 'Llama Mono', ui-monospace, monospace;
		display: grid;
		gap: 64px;
		width: min(72rem, calc(100% - 2rem));
		margin: 0 auto;
		padding: 40px 0 80px;
		color: var(--fg-default);
	}

	h1 {
		margin: 0;
		font-size: clamp(40px, 6vw, 64px);
		font-weight: 700;
		line-height: 1;
	}

	h2 {
		margin: 0;
		font-size: 28px;
		font-weight: 600;
		line-height: 1.15;
	}

	p {
		max-width: 62ch;
		margin: 12px 0 0;
		color: var(--fg-muted);
		font-size: 16px;
		line-height: 1.5;
	}

	.note {
		color: var(--fg-muted);
		font-size: 13px;
	}

	.feature {
		display: grid;
		grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
		gap: 40px;
		align-items: start;
	}

	.ladder {
		display: grid;
		gap: 0;
		margin: 24px 0 0;
		padding: 0;
		list-style: none;
	}

	.ladder li {
		display: grid;
		grid-template-columns: 4.5rem 1fr;
		padding-block: 10px;
		border-bottom: 1px solid var(--line-default);
		font-size: 14px;
	}

	.bits {
		color: var(--fg-default);
		font: 600 13px var(--mono);
	}

	.plates {
		display: grid;
		gap: 16px;
	}

	.plate {
		display: grid;
		gap: 16px;
		margin: 0;
		padding: clamp(20px, 4vw, 40px);
	}

	.plate.light {
		background: #fafafa;
		color: #111111;
	}

	.plate.dark {
		background: #111111;
		color: #fafafa;
	}

	.plate figcaption {
		font: 500 11px var(--mono);
		letter-spacing: 0.04em;
		opacity: 0.7;
	}

	.rows {
		display: grid;
		gap: 12px;
		margin-top: 24px;
	}

	.row {
		display: grid;
		grid-template-columns: minmax(10rem, 1fr) repeat(3, minmax(0, 1.4fr));
		gap: 16px;
		align-items: center;
		padding: 16px;
	}

	.row:not(:last-child) {
		border-bottom: 1px solid var(--line-default);
	}

	.label {
		display: grid;
		gap: 4px;
	}

	.name {
		font-size: 16px;
		font-weight: 600;
	}

	.tee {
		display: grid;
		gap: 6px;
		justify-items: center;
	}

	.guide {
		display: grid;
		grid-template-columns: minmax(0, 18rem) minmax(0, 1fr);
		gap: 40px;
		align-items: center;
		margin-top: 24px;
	}

	dl {
		display: grid;
		gap: 12px;
		margin: 0;
	}

	dl div {
		padding-bottom: 12px;
		border-bottom: 1px solid var(--line-default);
	}

	dt {
		font-weight: 600;
	}

	dd {
		margin: 4px 0 0;
		color: var(--fg-muted);
		font-size: 14px;
	}

	@media (width < 720px) {
		.feature,
		.guide {
			grid-template-columns: 1fr;
		}

		.row {
			grid-template-columns: repeat(3, minmax(0, 1fr));
			padding-inline: 0;
		}

		.label {
			grid-column: 1 / -1;
		}
	}
</style>
