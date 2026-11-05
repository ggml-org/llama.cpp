<!-- From the + of C++ to the Llama icon: the shapes of the origin animation, drawn into the blueprint's own drawing.
     The hero owns the clock (t) and the camera; this layer only draws what t asks for, and fades out as the
     blueprint's outlined parts take over. -->
<svelte:options namespace="svg" />

<script lang="ts">
	import { triangles } from '../blueprint';
	import { corners, type Pt, SIZE, SPLIT } from '../logo';
	import {
		BECOMES,
		BLUEPRINT,
		BLUEPRINT_STEP,
		clamp,
		GLYPH,
		inOutCubic,
		inOutQuart,
		lerp,
		LIGHTNING,
		MID,
		outCubic,
		progress,
		S,
		span,
		STEP,
		steps,
		WORD,
		WX,
		X,
		Y
	} from '../origin';

	let { t }: { t: number } = $props();

	const LEAN = 1 / Math.tan(Math.PI / 3); // 30° from upright: the shear that turns upright edges into 60° ones

	const [pCpp, pPlus, pBolt, pGrid, pSoft, pFill] = $derived(
		['C++', 'Plus', 'Bolt', 'Grid', 'Soften', 'Fill'].map((s) => progress(t, STEP[s]))
	);

	// the Grid step: the triangle grid the lean put the L on, across the whole frame, drawn line by line from the
	// icon outwards (nearest lines first), each over the last two fifths of the step
	const C: Pt = [SIZE / 2, SIZE / 2];
	const off = ([a, b]: [Pt, Pt]) =>
		Math.abs((b[0] - a[0]) * (a[1] - C[1]) - (a[0] - C[0]) * (b[1] - a[1])) /
		Math.hypot(b[0] - a[0], b[1] - a[1]);
	const GRID = triangles([-900, -500, 1500, 1300]).sort((p, q) => off(p) - off(q));
	const drawn = (i: number) => clamp((pGrid - (i / GRID.length) * 0.6) / 0.4);
	// the L step: the parts of the plus the L doesn't need fade out as the camera moves
	const gone_ = $derived(inOutCubic(clamp(progress(t, STEP.L) / 0.6)));
	// a line drawing, like the sketches, until the icon is finished; then it fills in, just before the blueprint
	// the blueprint's outlined parts take over from the solid icon
	const B0 = steps[BLUEPRINT_STEP].from;
	const gone = $derived(span(t, B0 + BLUEPRINT.parts[0], B0 + BLUEPRINT.parts[1]));

	// upright blocks, in upright coordinates (u across, y down), the same as the step tiles
	type Box = [number, number, number, number];
	const block: Record<string, Box> = {
		bottom: [-120, SIZE, 120, SIZE + SPLIT],
		centre: [-120, SPLIT, 120, SIZE],
		left: [-480, SPLIT, -120, SIZE],
		right: [120, SPLIT, 480, SIZE],
		top: [-120, 0, 120, SPLIT]
	};

	// the lean pivots on the baseline (y = SIZE), so the feet stay put while the tops slide right
	const shearPt = ([u, y]: Pt, k: number): Pt => [u + (SIZE - y) * k, y];
	const poly = ([u0, y0, u1, y1]: Box, k = 0) => {
		const pts = (
			[
				[u0, y0],
				[u1, y0],
				[u1, y1],
				[u0, y1]
			] as Pt[]
		).map((p) => shearPt(p, k));

		return `M${pts.map((p) => p.join(' ')).join('L')}Z`;
	};
	const lerpBox = (a: Box, b: Box, p: number) => a.map((v, i) => lerp(v, b[i], p)) as Box;

	// the middle + of "C++" as five blocks, each growing into the plus's own
	const glyphBox = (x0: number, x1: number, ylo: number, yhi: number): Box => [
		X + (MID + x0) * S,
		Y - yhi * S,
		X + (MID + x1) * S,
		Y - ylo * S
	];
	const fromGlyph: Record<string, Box> = {
		bottom: glyphBox(214, 328, 44, 231),
		centre: glyphBox(214, 328, 231, 345),
		left: glyphBox(27, 214, 231, 345),
		right: glyphBox(328, 515, 231, 345),
		top: glyphBox(214, 328, 345, 532)
	};
	const at = $derived(
		Object.fromEntries(
			Object.keys(block).map((b) => [b, lerpBox(fromGlyph[b], block[b], pPlus)])
		) as Record<string, Box>
	);
	// the C and the other + step aside and fade while the middle + grows
	const plusStep = steps[STEP.Plus];
	const aside = $derived(
		outCubic(clamp((t - plusStep.from) / ((plusStep.to - plusStep.from) * 0.7)))
	);
	// the plus's blocks take over from the middle + of "C++" just as it starts to grow
	const blocks = $derived(clamp((t - plusStep.from) / 0.2));

	// "llama.cpp": "llama." steps aside and fades while the camera closes in on ".cpp"; once it has settled, each p
	// cross-fades into a + as it slides the few units to the +'s place (the c already is the c of "c++"). Nothing is
	// resized, so no letter gets stretched
	const zoomStep = steps[STEP['.cpp']];
	const leave = $derived(
		outCubic(clamp((t - zoomStep.from) / ((zoomStep.to - zoomStep.from) * 0.6)))
	);
	const at_ = (x: number) => `translate(${x} ${Y}) scale(${S} ${-S})`;
	const letters = $derived(
		WORD.layout.map(([ch, x], i) => {
			const g = WORD.glyphs[ch],
				ox = WX + x * S,
				to = BECOMES[i];

			if (!to)
				return {
					d: g.d,
					i,
					opacity: 1 - leave,
					transform: `translate(${ox - 200 * leave} ${Y}) scale(${S} ${-S})`
				};

			const tox = X + to.x * S,
				now = lerp(ox, tox, pCpp);
			// C and the second + hand over to the "C++" drawn below once they land; the middle + to the plus's blocks
			const stays = i === 7 ? 1 - blocks : pCpp < 1 ? 1 : 0;

			if (to.to === 'C') return { d: g.d, i, opacity: stays, transform: at_(now) }; // the c just stays

			return {
				d: g.d,
				i,
				opacity: 1 - pCpp,
				to: { d: GLYPH[to.to], opacity: pCpp * stays, transform: at_(now) },
				transform: at_(now)
			};
		})
	);

	// the parts that go: the left and bottom arms and the centre they share with the L
	const going = $derived([at.left, at.bottom, at.centre]);

	// the finished blades with their outer corners rounded by s (0 sharp, 1 the brand's own curve)
	const towards = (a: Pt, b: Pt, s: number): Pt => [
		a[0] + (b[0] - a[0]) * s,
		a[1] + (b[1] - a[1]) * s
	];
	function blade(i: 0 | 1, s: number) {
		const c = corners[i];
		const e = towards(c.sharp, c.tangentEdge, s).join(' ');
		const f = towards(c.sharp, c.tangentFlat, s).join(' ');
		const r = c.r * s;

		return i === 0
			? `M0 ${SPLIT}L${e}A${r} ${r} 0 0 1 ${f}L466.154 0L240 ${SPLIT}Z`
			: `M240 ${SPLIT}L600 ${SPLIT}L${e}A${r} ${r} 0 0 1 ${f}L120 ${SIZE}Z`;
	}
	// the strike, in three beats over the Bolt step (x = 0 to 1): the bolt appears over the upright L and strikes
	// it twice, the L in its own colour (the idea); both fade out and the stage is empty a moment;
	// then they come back as the slanted L inside the bolt, flashing and glitching, before the bolt fades and the L stays
	const DARK = 0.34,
		GONE = 0.28,
		SETTLE = 0.57,
		SNAP = 0.38,
		UPRIGHT = 0.24;
	const leaned = $derived(pBolt >= SNAP ? 1 : 0);
	const k = $derived(LEAN * leaned);

	// the strike: a lightning bolt fitted to the L races down over it from the top, flares, flickers once and dies
	// away in yellow, with a glow; the L itself keeps its shape and only catches a little of the light
	// the Soften step: a dashed wheel grows into each outer corner, turning; as it slows to a stop the corner rounds
	// onto it, and it comes to rest with its spoke on the point where the curve meets the flat edge
	const wheel = $derived(outCubic(clamp(pSoft / 0.3)));
	const turn = $derived(-200 * (1 - outCubic(pSoft)));
	const round = $derived(inOutQuart(clamp((pSoft - 0.3) / 0.7)));
	// the bolt comes first, as the idea: it strikes where the slanted L will be, and the slanted L appears with it;
	// the two flash together a few times (on, off, on, dim, on), then the bolt settles to a glow and fades once the
	// L has held there a moment
	const leanEnd = steps[STEP.Lean].to;
	const glow = 0.7;
	// [until, level]: the upright L's flashes, then the slanted L's
	const uprightPulses: [number, number][] = [
		[0.1, 1],
		[0.15, 0.1],
		[UPRIGHT, 1]
	]; // two strokes, no more
	const pulses: [number, number][] = [
		[0.42, 1],
		[0.45, 0.08],
		[0.49, 1],
		[0.52, 0.3],
		[SETTLE, 1]
	];
	const strike = $derived.by(() => {
		const x = pBolt;

		if (x <= 0) return 0;

		if (x < 0.04) return x / 0.04; // the bolt appears over the upright L

		const up = uprightPulses.find(([until]) => x < until);

		if (up) return up[1]; // and flashes with it

		if (x < GONE) return lerp(1, 0.8, (x - UPRIGHT) / (GONE - UPRIGHT));

		if (x < DARK) return lerp(0.8, 0, (x - GONE) / (DARK - GONE)); // and goes, with the L

		if (x < SNAP) return 0; // a moment of nothing

		const pulse = pulses.find(([until]) => x < until);

		if (pulse) return pulse[1]; // back as the slanted L in the bolt, flashing

		if (x < 1) return lerp(1, glow, outCubic((x - SETTLE) / (1 - SETTLE)));

		return glow * (1 - outCubic(clamp((t - leanEnd - 0.05) / 0.3)));
	});
	const reach = $derived(-20 + 1100 * outCubic(clamp(pBolt / 0.06))); // how far down the strike has got: past the tip (y 784) and its glow

	// the L flashes with the bolt, so the bolt seems to turn into it: its strokes glow in their own colour, and while the
	// strike is fresh they glitch, two offset copies jumping every ~33 ms like a broken signal. It is all driven by
	// t, so scrubbing shows the same glitch every time
	const rand = (n: number) => {
		const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453;

		return x - Math.floor(x);
	};
	const glitch = $derived(
		pBolt >= SNAP && pBolt < 0.66 ? Math.sqrt(1 - (pBolt - SNAP) / (0.66 - SNAP)) : 0
	); // strong at first, easing off
	// the L's own visibility: the upright one stays under the bolt, then fades out with it; the slanted one blinks
	// with the flashes (dark between them), then stays
	// light mode only: the sky goes dark for the strike, like lightning at night, and day comes back as the bolt
	// fades; meanwhile the L is drawn in the night's light colour (--l-night is 1 in light mode, 0 in dark)
	const night = $derived(
		pBolt <= 0
			? 0
			: t < leanEnd + 0.2
				? clamp(pBolt / 0.03)
				: 1 - inOutCubic(clamp((t - leanEnd - 0.2) / 0.35))
	);
	const lAlpha = $derived(
		pBolt <= 0.04
			? 1
			: pBolt < UPRIGHT
				? Math.max(0.15, strike)
				: pBolt < GONE
					? 1
					: pBolt < DARK
						? 1 - (pBolt - GONE) / (DARK - GONE)
						: pBolt < SNAP
							? 0
							: pBolt < SETTLE
								? Math.max(0.12, strike)
								: 1
	);
	const jitter = $derived.by(() => {
		if (!glitch) return null;

		const frame = Math.floor(t * 30);
		const r = [0, 1, 2, 3, 4, 5, 6].map((j) => rand(frame * 7 + j));

		return {
			a: `translate(${(r[0] - 0.5) * 100 * glitch} ${(r[1] - 0.5) * 34 * glitch})`,
			b: `translate(${(0.5 - r[2]) * 100 * glitch} ${(r[3] - 0.5) * 34 * glitch})`,
			bolt: `translate(${(r[0] - 0.5) * -36 * glitch} 0)`,
			c: `translate(${(r[5] - 0.5) * 60 * glitch} ${(0.5 - r[6]) * 24 * glitch})`,
			on: r[4] > 0.25 // some frames drop out, so it stutters
		};
	});
	const top = $derived(round > 0 ? blade(0, round) : poly(at.top, k));
	const foot = $derived(round > 0 ? blade(1, round) : poly(at.right, k));
</script>

{#if gone < 1}
	<g style:--night={night} style:--solid={pFill} style:opacity={1 - gone} class="morph">
		<defs>
			<filter
				filterUnits="userSpaceOnUse"
				height="1600"
				id="strike-glow"
				width="1400"
				x="-400"
				y="-400"
			>
				<feGaussianBlur stdDeviation="22" />
			</filter>

			<filter
				filterUnits="userSpaceOnUse"
				height="2000"
				id="strike-halo"
				width="1800"
				x="-600"
				y="-600"
			>
				<feGaussianBlur stdDeviation="64" />
			</filter>

			<clipPath id="strike-reach"
				><rect height={Math.max(0, reach + 40)} width="1400" x="-400" y="-40" /></clipPath
			>
		</defs>

		<!-- the night sky for the strike (light mode): a dark field under the whole drawing -->
		{#if night > 0}<rect class="night" height="14000" width="14000" x="-6000" y="-6000" />{/if}
		<!-- "llama.cpp", becoming "C++" -->
		{#if blocks < 1}
			{#each letters as l (l.i)}
				{#if l.opacity > 0}<path
						style:opacity={l.opacity}
						class="fill"
						d={l.d}
						transform={l.transform}
					/>{/if}

				{#if l.to && l.to.opacity > 0}<path
						style:opacity={l.to.opacity}
						class="fill"
						d={l.to.d}
						transform={l.to.transform}
					/>{/if}
			{/each}
		{/if}
		<!-- the rest of "C++": the C and the second + step aside and fade -->
		{#if pCpp >= 1 && aside < 1}
			<g style:opacity={1 - aside}>
				<path
					class="fill"
					d={GLYPH.C}
					transform="translate({X - 160 * aside} {Y}) scale({S} {-S})"
				/>

				<path
					class="fill"
					d={GLYPH.plus}
					transform="translate({X +
						(MID + GLYPH.advance.plus) * S +
						160 * aside} {Y}) scale({S} {-S})"
				/>
			</g>
		{/if}

		<g style:opacity={blocks}>
			<!-- the parts that go simply fade, under the L's own outline, so the edges it shares with them stay -->
			{#if gone_ < 1}
				{#each going as b, i (i)}<path style:opacity={1 - gone_} class="fill" d={poly(b)} />{/each}
			{/if}

			{#if pGrid > 0 && pFill < 1}
				<g style:opacity={1 - pFill} class="grid">
					{#each GRID as [a, b], i (i)}
						<line
							pathLength="1"
							stroke-dasharray="1"
							stroke-dashoffset={1 - drawn(i)}
							x1={a[0]}
							x2={b[0]}
							y1={a[1]}
							y2={b[1]}
						/>
					{/each}
				</g>
			{/if}

			<path style:opacity={lAlpha} class="blade" d={top} />

			<path style:opacity={lAlpha} class="blade" d={foot} />

			{#if wheel > 0}
				{#each corners as c, i (i)}
					{@const [cx, cy] = c.centre}
					<g
						style:opacity={wheel}
						class="wheel"
						transform="translate({cx} {cy}) rotate({i ? -turn : turn}) scale({0.6 +
							0.4 * wheel}) translate({-cx} {-cy})"
					>
						<circle class="rim" {cx} {cy} r={c.r} />

						<line class="spoke" x1={cx} x2={c.tangentFlat[0]} y1={cy} y2={c.tangentFlat[1]} />

						<circle class="hub" {cx} {cy} r="5" />
					</g>
				{/each}
			{/if}

			{#if jitter?.on}
				<g style:opacity={glitch} class="glitch">
					<g class="orange" transform={jitter.a}><path d={top} /><path d={foot} /></g>

					<g class="blue" transform={jitter.b}><path d={top} /><path d={foot} /></g>

					<g class="yellow" transform={jitter.c}><path d={top} /><path d={foot} /></g>
				</g>
			{/if}
		</g>
		<!-- the strike, over the L: a blurred yellow flare and the bolt itself, revealed from the top down -->
		{#if strike > 0}
			<g style:opacity={strike} clip-path="url(#strike-reach)">
				<g transform={jitter?.on ? jitter.bolt : undefined}>
					<!-- a blue halo, a yellow glow, and the white-hot bolt itself -->
					<path class="halo" d={LIGHTNING} filter="url(#strike-halo)" />

					<path class="flare" d={LIGHTNING} filter="url(#strike-glow)" />

					<path class="bolt" d={LIGHTNING} />
				</g>
			</g>
			<!-- the slanted L's own glow, in the bolt's yellow -->
			<g style:opacity={strike * lAlpha} class="l-flare" filter="url(#strike-glow)">
				<path d={top} />

				<path d={foot} />
			</g>
		{/if}
	</g>
{/if}

<style>
	.morph {
		pointer-events: none;
	}
	/* outlines in the icon's own colour, filled only once the icon is finished (--solid) */
	/* in the night of a light-mode strike, the L takes the night's light colour so it reads against it */
	.morph {
		--l-lit: color-mix(
			in srgb,
			var(--l-night-ink) calc(var(--night, 0) * var(--l-night) * 100%),
			var(--l-shape)
		);
	}
	.night {
		fill: var(--l-night-sky);
		opacity: calc(var(--night, 0) * var(--l-night));
	}
	.fill,
	.blade {
		fill: var(--l-shape);
		fill-opacity: var(--solid, 0);
		stroke: var(--l-lit);
		stroke-width: 2;
		stroke-linejoin: round;
		vector-effect: non-scaling-stroke;
	}
	/* the L stays in its own colour (white on dark, black on light) and glows in it; the colour is in the glitch */
	.l-flare path {
		/* a black glow reads as a smudge on light, so the theme can soften it; at night it glows in full */
		opacity: calc(
			var(--l-ink-glow, 1) + (1 - var(--l-ink-glow, 1)) * var(--night, 0) * var(--l-night)
		);
		fill: none;
		stroke: var(--l-lit);
		stroke-width: 16;
		stroke-linejoin: round;
	}
	/* the glitch copies split into the brand orange and the bolt's yellow and blue */
	.glitch path {
		fill: none;
		stroke-width: 3;
		vector-effect: non-scaling-stroke;
	}
	.glitch .orange path {
		stroke: var(--l-bolt);
	}
	.glitch .blue path {
		stroke: var(--l-bolt-halo);
		opacity: 0.85;
	}
	.glitch .yellow path {
		stroke: var(--l-bolt-glow);
		opacity: 0.8;
	}
	/* the Grid step's triangles: faint, in the L's own colour (which is light in a light-mode night) */
	/* in drawing units, not non-scaling, so the draw-on dash (pathLength 1) runs the whole line: about 1px in this frame */
	.grid line {
		stroke: var(--l-lit);
		stroke-opacity: 0.22;
		stroke-width: 2.6;
	}
	/* the Soften step's wheels: dashed, so their turning shows */
	.wheel .rim,
	.wheel .spoke {
		fill: none;
		stroke: var(--l-ink);
		stroke-opacity: 0.55;
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
	}
	.wheel .rim {
		stroke-dasharray: 7 5;
	}
	.wheel .hub {
		fill: var(--l-ink);
		fill-opacity: 0.55;
	}
	/* the bolt: a white-hot core in a yellow glow and a blue halo, like real lightning */
	.halo {
		fill: var(--l-bolt-halo);
		opacity: 0.75;
	}
	.flare {
		fill: var(--l-bolt-glow);
		opacity: 0.9;
	}
	.bolt {
		fill: var(--l-bolt-core);
	}
</style>
