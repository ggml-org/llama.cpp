<!--
	A sky quantised: the day or night sky's gradient as a grid of flat squares, each the gradient's colour at its
	centre snapped to a few levels and nudged a little, as a low-bit quant keeps a model's weights. One square is
	cinder, and the squares round it warm towards cinder light, the art colour, the nearer the warmer. The references'
	pixelated field, drawn from llama.cpp's own idea.
-->
<script lang="ts">
	interface Props {
		/** squares across; the rows follow the field's shape */
		columns?: number;
		rows?: number;
		seed?: number;
		sky?: 'day' | 'night';
	}

	let { columns = 12, rows = 6, seed = 11, sky = 'day' }: Props = $props();

	// the skies' stops (css/colors.css in llama-brand), top to bottom
	const STOPS = {
		day: ['#6fa6dc', '#8fbde5', '#c4ddf6', '#e8f2fc'],
		night: ['#0f1a2b', '#13213a', '#1b3659', '#2e4668']
	};

	const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

	function at(t: number) {
		const s = STOPS[sky].map(hex);
		const x = Math.min(0.999, Math.max(0, t)) * (s.length - 1);
		const i = Math.floor(x);
		const f = x - i;

		return s[i].map((c, k) => Math.round(c + (s[i + 1][k] - c) * f));
	}

	const accent = $derived(columns * Math.floor(rows * 0.6) + Math.floor(columns * 0.7));

	// cinder light, which the squares near the cinder one blend towards
	const WARM = hex('#f3c1ab');

	const squares = $derived.by(() => {
		let r = seed;

		const rand = () => (r = (r * 1103515245 + 12345) % 2147483648) / 2147483648;
		const ax = accent % columns;
		const ay = Math.floor(accent / columns);

		return Array.from({ length: columns * rows }, (_, i) => {
			const cx = i % columns;
			const cy = Math.floor(i / columns);
			// a diagonal sky, deeper at the top left, with a nudge of noise snapped to quarter steps
			const t =
				Math.round(
					((cy / (rows - 1)) * 0.75 + (cx / (columns - 1)) * 0.25 + (rand() - 0.5) * 0.3) * 4
				) / 4;
			// the warmth, by distance from the cinder square, snapped to thirds
			const d = Math.hypot(cx - ax, (cy - ay) * 1.2);
			const w = Math.round(Math.max(0, 1 - d / (columns * 0.35)) * 3) / 3;
			const [cr, cg, cb] = at(t).map((c, k) =>
				Math.round(c + (WARM[k] - c) * w * (sky === 'day' ? 0.9 : 0.55))
			);

			return `rgb(${cr} ${cg} ${cb})`;
		});
	});
</script>

<div style:--c={columns} style:--r={rows} aria-hidden="true" class="quant">
	{#each squares as fill, i (i)}
		<i style:background={i === accent ? 'var(--art-cinder)' : fill}></i>
	{/each}
</div>

<style>
	.quant {
		display: grid;
		grid-template-columns: repeat(var(--c), 1fr);
		grid-template-rows: repeat(var(--r), 1fr);
		width: 100%;
		height: 100%;
	}
</style>
