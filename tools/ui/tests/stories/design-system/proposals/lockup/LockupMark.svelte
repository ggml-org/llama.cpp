<!--
	One lockup of the proposal at a given step and size: the icon (drawn, or the real U+E000 glyph at the glyph step) and
	the wordmark set live in Llama Sans Bold, .cpp in cinder. size is the wordmark's font size in px; the icon is centred
	on the capitals.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../../../brand/logo-showcase/lib/logo';
	import { CAP, FIRST_LSB, type Step } from './geometry';

	let {
		ink,
		name,
		size,
		step
	}: { ink: string; name: 'Llama' | 'llama.cpp'; size: number; step: Step } = $props();

	const GLYPH = ' ';

	const side = $derived(step.icon * CAP);
	const top = $derived(Math.min(-CAP / 2 - side / 2, -CAP - 60));
	const bottom = $derived(Math.max(-CAP / 2 + side / 2, 230));
	const textX = $derived(step.glyph ? 0 : side + step.gap * CAP - FIRST_LSB[name]);

	let width = $state(4000);

	function measure(node: SVGTextElement) {
		const update = () => {
			const box = node.getBBox();

			width = box.x + box.width + 20;
		};

		document.fonts.ready.then(update);

		return { update };
	}
</script>

<svg
	style:width="{(width * size) / 1000}px"
	aria-label={name}
	class="block h-auto max-w-full shrink-0"
	role="img"
	viewBox="0 {top} {width} {bottom - top}"
>
	{#if !step.glyph}
		<g fill={ink} transform="translate(0 {-CAP / 2 - side / 2}) scale({side / SIZE})">
			{#each PATHS as d (d)}<path {d} />{/each}
		</g>
	{/if}

	<text
		use:measure
		fill={ink}
		font-family="'Llama Sans'"
		font-size="1000"
		font-weight="700"
		x={textX}
		y="0"
		>{step.glyph ? GLYPH : ''}{name.slice(0, 5)}<tspan fill="var(--art-cinder)"
			>{name.slice(5)}</tspan
		></text
	>
</svg>
