<!--
	The Llama icon's blueprint, with its origin animation drawn into it: the + of C++ grows into the icon and settles
	into this very drawing. Hover, tap or pick any part to highlight the idea it belongs to.
	t: where the origin animation is (END is the finished blueprint); autoplay plays it once on load.
	pinned: the part a tap or a chip has highlighted. controls: the floating play bar, chips and caption.
-->
<script lang="ts">
	import { box, deg, dims, type Feature, features, lines, r, vb } from '../blueprint';
	import { corners, PATHS, SIZE, SPLIT } from '../logo';
	import {
		BLUEPRINT,
		BLUEPRINT_STEP,
		camera,
		END,
		span,
		stepAt,
		steps as originSteps,
		TILES
	} from '../origin';
	import Morph from './Morph.svelte';
	import { onMount } from 'svelte';

	let {
		autoplay = true,
		controls = true,
		pinned = $bindable(null),
		t = $bindable(0)
	}: { t?: number; pinned?: Feature | null; autoplay?: boolean; controls?: boolean } = $props();

	let hovered = $state<Feature | null>(null);
	const active = $derived(hovered ?? pinned);

	// the origin animation's clock
	let playing = $state(false);
	const done = $derived(t >= END);
	let frame = 0;
	let last = 0;
	function tick(now: number) {
		t = Math.min(END, t + (now - last) / 1000);
		last = now;

		if (t < END && playing) frame = requestAnimationFrame(tick);
		else playing = false;
	}
	function play() {
		hovered = pinned = null;

		if (t >= END) t = 0;

		playing = true;
		last = performance.now();
		frame = requestAnimationFrame(tick);
	}
	function pause() {
		playing = false;
		cancelAnimationFrame(frame);
	}
	const seek = (to: number) => {
		pause();
		t = to;
	};
	onMount(() => {
		// with reduced motion, it opens on the finished blueprint and the animation waits for the play button
		if (autoplay) {
			if (matchMedia('(prefers-reduced-motion: reduce)').matches) t = END;
			else document.fonts.ready.then(() => setTimeout(play, 250));
		}

		return pause;
	});

	// the camera ends on the blueprint's own frame, so the last frame of the animation is the blueprint as drawn
	const final: [number, number, number, number] = [
		(box[0] + box[2]) / 2,
		(box[1] + box[3]) / 2,
		box[2] - box[0],
		box[3] - box[1]
	];
	const view = $derived(done ? vb : camera(t, final[2] / final[3], final));
	// the blueprint's layers fade in one after another as the animation hands over
	const B0 = originSteps[BLUEPRINT_STEP].from;
	const fade = $derived(
		Object.fromEntries(
			Object.entries(BLUEPRINT).map(([k, [a, b]]) => [k, done ? 1 : span(t, B0 + a, B0 + b)])
		) as Record<keyof typeof BLUEPRINT, number>
	);
	const step = $derived(stepAt(t));
	// the steps that have a tile below carry its number (01 llama.cpp to 06 Soften)
	const stepNumber = (i: number) => {
		const n = TILES.indexOf(originSteps[i].name);

		return n < 0 ? '' : `${String(n + 1).padStart(2, '0')} `;
	};
	const current = $derived(features.find((x) => x.f === active));
	const featureAt = (e: Event) =>
		((e.target as Element).closest('[data-f]') as HTMLElement | null)?.dataset.f as
			| Feature
			| undefined;
	const pin = (f: Feature) => (pinned = pinned === f ? null : f);
</script>

<section aria-label="Construction" class="hero">
	<div class="hero-stage">
		<!-- pointer and tap highlighting is a shortcut; the legend buttons below give the same to keyboards and screen readers -->
		<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
		<svg
			class:animating={!done}
			class:focused={done && active}
			aria-label={done
				? 'Blueprint of the Llama icon with its construction lines and dimensions'
				: `Animation: the + of C++ becomes the Llama icon and then its blueprint. Now: ${originSteps[step].name}`}
			onclick={(e) => {
				if (!done) return;

				const f = featureAt(e);

				if (f) pin(f);
				else pinned = null;
			}}
			onpointerleave={() => (hovered = null)}
			onpointerover={(e) => {
				if (!done) return;

				if (e.pointerType === 'mouse') hovered = featureAt(e) ?? null;
			}}
			role="img"
			viewBox={view}
		>
			<defs>
				<pattern height="20" id="bp-minor" patternUnits="userSpaceOnUse" width="20">
					<path class="grid-minor" d="M20 0H0V20" />
				</pattern>

				<pattern height="120" id="bp-major" patternUnits="userSpaceOnUse" width="120">
					<path class="grid-major" d="M120 0H0V120" />
				</pattern>
			</defs>

			<g style:opacity={fade.grid}>
				<rect
					fill="url(#bp-minor)"
					height={box[3] - box[1]}
					width={box[2] - box[0]}
					x={box[0]}
					y={box[1]}
				/>

				<rect
					fill="url(#bp-major)"
					height={box[3] - box[1]}
					width={box[2] - box[0]}
					x={box[0]}
					y={box[1]}
				/>
			</g>

			<g style:opacity={fade.lines}>
				{#each lines as { a, b, f }, i (i)}
					<g class:on={f && active === f} class:static={!f} data-f={f}>
						<line class="guide" x1={a[0]} x2={b[0]} y1={a[1]} y2={b[1]} />

						<line class="hit" x1={a[0]} x2={b[0]} y1={a[1]} y2={b[1]} />
					</g>
				{/each}

				{#each corners as c, i (i)}
					<g class:on={active === 'corners'} data-f="corners">
						<circle class="guide" cx={c.centre[0]} cy={c.centre[1]} r={c.r} />

						<circle class="hit" cx={c.centre[0]} cy={c.centre[1]} r={c.r} />

						<circle class="joint" cx={c.centre[0]} cy={c.centre[1]} r="4" />
					</g>
				{/each}
			</g>

			<g style:opacity={fade.parts}>
				{#each [['lower', PATHS[0]], ['upper', PATHS[1]]] as [f, d] (f)}
					<path class:on={active === f} class="part" {d} data-f={f} />
				{/each}
			</g>
			<!-- the origin animation, drawn into this same drawing until the blueprint takes over -->
			{#if !done}<Morph {t} />{/if}

			<!-- dimensions -->
			<g style:opacity={fade.dims}>
				{#each dims as d, i (i)}
					<g class:on={d.f && active === d.f} class:static={!d.f} class="dimension" data-f={d.f}>
						<line x1={d.a[0]} x2={d.b[0]} y1={d.a[1]} y2={d.b[1]} />

						<line class="hit" x1={d.a[0]} x2={d.b[0]} y1={d.a[1]} y2={d.b[1]} />

						{#each [d.a, d.b] as p, j (j)}
							<line x1={p[0] - d.t[0]} x2={p[0] + d.t[0]} y1={p[1] - d.t[1]} y2={p[1] + d.t[1]} />
						{/each}

						<text
							text-anchor="middle"
							transform={d.rot ? `rotate(-90 ${d.at[0]} ${d.at[1]})` : undefined}
							x={d.at[0]}
							y={d.at[1]}>{d.label}</text
						>
					</g>
				{/each}

				<g class:on={active === 'angle'} data-f="angle">
					<path class="dimension-arc" d="M60 {SPLIT} A60 60 0 0 0 30 {SPLIT - 52}" />

					<text class="bp-label" x="72" y={SPLIT - 18}>{deg}°</text>
				</g>

				{#each corners as c, i (i)}
					<text
						class:on={active === 'corners'}
						class="bp-label"
						data-f="corners"
						text-anchor={i ? 'start' : 'end'}
						x={c.centre[0] + (i ? 22 : -22)}
						y={c.centre[1] + (i ? 30 : -18)}>R {r}</text
					>
				{/each}
			</g>

			<!-- title block -->
			<g style:opacity={fade.title} class="title-block" transform="translate(620 600)">
				<rect height="140" width="280" />

				<line x1="0" x2="280" y1="44" y2="44" />

				<line x1="0" x2="280" y1="92" y2="92" />

				{#each [93, 186] as x (x)}<line x1={x} x2={x} y1="92" y2="140" />{/each}

				<text class="bp-title" x="12" y="30">LLAMA · ICON</text>

				<text class="bp-label" x="12" y="74">PLUS + L → BOLT</text>

				{#each [['ANGLE', `${Math.round(+deg)}°`, 'angle'], ['GRID', '120', 'steps'], ['SPLIT', `${SPLIT}:${SIZE - SPLIT}`, 'split']] as [k, val, f], n (k)}
					<g class:on={active === f} data-f={f}>
						<rect class="hit-area" height="48" width="93" x={n * 93} y="92" />

						<text class="bp-small" x={n * 93 + 10} y="112">{k}</text>

						<text class="bp-label" x={n * 93 + 10} y="130">{val}</text>
					</g>
				{/each}
			</g>
		</svg>
		<!-- the controls, the highlight chips and the caption: one panel floating over the bottom of the drawing -->
		{#if controls}
			<div class="legend">
				<div class="bar">
					<button
						aria-label={playing ? 'Pause' : done ? 'Replay the origin' : 'Play'}
						class="play"
						onclick={() => (playing ? pause() : play())}
						type="button"
					>
						{#if playing}
							<svg aria-hidden="true" viewBox="0 0 16 16"><path d="M4 3h3v10H4zM9 3h3v10H9z" /></svg
							>
						{:else if done}
							<svg aria-hidden="true" viewBox="0 0 16 16"
								><path d="M8 3a5 5 0 1 0 5 5h-1.6A3.4 3.4 0 1 1 8 4.6V7l3.5-3.2L8 .6z" /></svg
							>
						{:else}
							<svg aria-hidden="true" viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z" /></svg>
						{/if}
					</button>

					<input
						aria-label="Position in the origin animation"
						max={END}
						min="0"
						oninput={(e) => seek(+e.currentTarget.value)}
						step="0.01"
						type="range"
						value={t}
					/>
				</div>

				<div
					class:idle={!done}
					aria-label="Highlight a part of the drawing"
					class="chips"
					inert={!done}
					role="group"
				>
					{#each features as x (x.f)}
						<button
							class:lit={active === x.f}
							aria-pressed={pinned === x.f}
							onblur={() => (hovered = null)}
							onclick={() => pin(x.f)}
							onfocus={() => (hovered = x.f)}
							onpointerenter={(e) => e.pointerType === 'mouse' && (hovered = x.f)}
							onpointerleave={() => (hovered = null)}
							type="button">{x.name}</button
						>
					{/each}
				</div>

				<p aria-live="polite" class="caption">
					{#if !done}<strong>{stepNumber(step)}{originSteps[step].name}.</strong>
						{originSteps[step].note}{:else if current}<strong>{current.name}.</strong>
						{current.note}{:else}Hover or tap any part of the drawing, or pick one here.{/if}
				</p>
			</div>
		{/if}
	</div>
</section>

<style>
	.hero {
		border: 1px solid var(--l-rule);
		border-radius: 12px;
		overflow: hidden;
		background: var(--l-paper);
	}
	.grid-minor {
		fill: none;
		stroke: var(--l-grid);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.grid-major {
		fill: none;
		stroke: var(--l-grid-major);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.part {
		fill: var(--l-faint);
		stroke: var(--l-ink);
		stroke-width: 2;
		vector-effect: non-scaling-stroke;
	}
	.dimension line,
	.dimension-arc,
	.title-block rect,
	.title-block line {
		fill: none;
		stroke: var(--l-muted);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.dimension text,
	.bp-label,
	.bp-small,
	.bp-title {
		font: 500 15px var(--mono);
		letter-spacing: 0.06em;
		fill: var(--l-muted);
	}
	.bp-small {
		font-size: 11px;
	}
	.bp-title {
		font:
			700 20px 'Llama Sans',
			var(--body);
		letter-spacing: 0;
		fill: var(--l-ink);
	}
	/* blueprint highlighting: with a part active, everything tagged dims except that part's idea */
	[data-f] {
		cursor: pointer;
		transition: opacity 120ms ease-out;
	}
	[data-f] :is(line, circle, path, text),
	[data-f]:is(path, text) {
		transition:
			stroke 120ms ease-out,
			fill 120ms ease-out,
			fill-opacity 120ms ease-out,
			stroke-width 120ms ease-out;
	}
	.static {
		transition: opacity 120ms ease-out;
	}
	.hero svg.focused [data-f]:not(.on),
	.hero svg.focused .static {
		opacity: 0.14;
	}
	.hit {
		fill: none;
		stroke: transparent;
		stroke-width: 16;
		vector-effect: non-scaling-stroke;
		pointer-events: stroke;
	}
	.hit-area {
		fill: transparent;
	}
	.on > .guide {
		stroke: var(--l-ink);
		stroke-width: 1.6;
		stroke-dasharray: none;
	}
	.part.on {
		fill: var(--l-shape-2);
		fill-opacity: 0.6;
		stroke-width: 2.5;
	}
	.on .joint {
		fill: var(--l-ink);
	}
	.dimension.on line,
	.on .dimension-arc {
		stroke: var(--l-ink);
	}
	.dimension.on text,
	.on .bp-label,
	.on .bp-small,
	.bp-label.on {
		fill: var(--l-ink);
	}
	.hero svg.animating {
		cursor: default;
	}
	/* the drawing and its panel (controls, chips, caption), which floats along the bottom of the drawing and shows
	   while the pointer is over it or a control has focus */
	.hero-stage {
		position: relative;
	}
	.bar {
		display: grid;
		grid-template-columns: auto 1fr;
		align-items: center;
		gap: 12px;
	}
	.play {
		display: grid;
		place-items: center;
		width: 32px;
		height: 32px;
		padding: 0;
		border: 0;
		border-radius: 50%;
		background: var(--l-ink);
		color: var(--l-bg);
		cursor: pointer;
	}
	.play svg {
		width: 13px;
		height: 13px;
		fill: currentColor;
	}
	.play:focus-visible {
		outline: 2px solid var(--l-ink);
		outline-offset: 2px;
	}
	.bar input[type='range'] {
		width: 100%;
		accent-color: var(--l-ink);
	}
	.chips.idle {
		opacity: 0.4;
	}
	.chips {
		transition: opacity 300ms ease-out;
	}
	.legend {
		position: absolute;
		left: 12px;
		right: 12px;
		bottom: 12px;
		display: grid;
		gap: 10px;
		padding: 12px 14px 14px;
		border: 1px solid var(--l-rule);
		border-radius: 12px;
		background: color-mix(in srgb, var(--l-paper) 82%, transparent);
		backdrop-filter: blur(12px);
		-webkit-backdrop-filter: blur(12px);
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
		opacity: 0;
		transform: translateY(6px);
		pointer-events: none;
		transition:
			opacity 180ms ease-out,
			transform 180ms ease-out;
	}
	.hero-stage:hover .legend,
	.hero-stage:focus-within .legend {
		opacity: 1;
		transform: none;
		pointer-events: auto;
	}
	/* touch screens can't hover, and on a phone a floating panel would cover most of the drawing: there the panel
	   sits below the drawing instead */
	@media (hover: none) {
		.legend {
			position: static;
			opacity: 1;
			transform: none;
			pointer-events: auto;
			border: 0;
			border-top: 1px solid var(--l-rule);
			border-radius: 0;
			background: none;
			backdrop-filter: none;
			-webkit-backdrop-filter: none;
			box-shadow: none;
		}
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.chips button {
		padding: 0.35em 0.8em;
		border: 0;
		border-radius: 999px;
		background: none;
		box-shadow: inset 0 0 0 1px var(--l-rule);
		color: var(--l-muted);
		font:
			500 13px / normal 'Llama Sans',
			var(--body);
		cursor: pointer;
		transition:
			background-color 160ms ease,
			color 160ms ease;
	}
	.chips button.lit {
		box-shadow: inset 0 0 0 1px var(--l-ink);
		color: var(--l-ink);
	}
	.chips button[aria-pressed='true'] {
		background: var(--l-ink);
		color: var(--l-bg);
	}
	.chips button:focus-visible {
		outline: 2px solid var(--l-ink);
		outline-offset: 2px;
	}
	.caption {
		margin: 0;
		min-height: 2.9em;
		font:
			400 14px / 1.45 'Llama Sans',
			var(--body);
		color: var(--l-muted);
	}
	.caption strong {
		color: var(--l-ink);
		font-weight: 600;
	}
	.hero svg {
		display: block;
		width: 100%;
		height: auto;
		max-height: 80vh;
	}
	.guide {
		stroke: var(--l-line);
		stroke-width: 1;
		stroke-dasharray: 6 4;
		fill: none;
		vector-effect: non-scaling-stroke;
	}
	.joint {
		fill: var(--l-line);
	}
	.guide {
		stroke: var(--l-line);
		stroke-width: 1;
		stroke-dasharray: 6 4;
		fill: none;
		vector-effect: non-scaling-stroke;
	}
	.joint {
		fill: var(--l-line);
	}
</style>
