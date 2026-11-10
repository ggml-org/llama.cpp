<!--
	Pi's Virtual Models, after Julien's post and in its layout: the title centred at the top, Pi on the left with the
	model picked in it, a line from it to each model it can send a query to (where it runs in a Mono label, its name as
	a headline) and, at the right, what runs that model. Pi's and Hugging Face's marks are their own files (from
	pi.dev/press-kit and huggingface.co/brand), never recoloured; llama.cpp's is its lockup, typed in the ground's ink.
	Each mark keeps half its height clear all round. The local route is the cinder detail.
-->
<script lang="ts">
	import hfMark from './logos/hugging-face.svg?url';
	import piMark from './logos/pi.svg?url';
	import { FORMATS, LOCKUP, type VirtualModelsCopy } from './post';
	import PostFrame from './PostFrame.svelte';

	interface Props {
		copy: VirtualModelsCopy;
		overlay?: boolean;
		tone: 'dark' | 'light';
	}

	let { copy, overlay = false, tone }: Props = $props();

	const rows = $derived(copy.routes.length);
</script>

<PostFrame format={FORMATS.landscape} label="Virtual models, {tone}" {overlay} {tone}>
	<div class="piece">
		<h2 class="t-display title">{copy.title}</h2>

		<div style:--rows={rows} class="diagram">
			<div class="from">
				<!-- Pi's file draws its mark in the middle 59% of its square: cropped to the mark, so it sizes and
				     aligns by what shows -->
				<span class="pi"><img alt="Pi" src={piMark} /></span>

				<p class="caption">
					<span class="name">{copy.from.name}</span>

					<span class="id">{copy.from.model}</span>
				</p>
			</div>

			<div class="links">
				<svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 100 {rows * 100}">
					{#each copy.routes as route, i (route.name)}
						<path
							class:accent={i === 0}
							d="M0 {rows * 50} C50 {rows * 50} 50 {i * 100 + 50} 100 {i * 100 + 50}"
							vector-effect="non-scaling-stroke"
						/>
					{/each}
				</svg>

				<i style:left="0" style:top="50%" class="dot"></i>

				{#each copy.routes as route, i (route.name)}
					<i style:left="100%" style:top="{((i + 0.5) / rows) * 100}%" class="dot"></i>
				{/each}
			</div>

			<ol class="routes">
				{#each copy.routes as route, i (route.name)}
					<li>
						<div class="model">
							<p class="t-label"><i class={['tick', { accent: i === 0 }]}></i>{route.label}</p>

							<p class="t-headline">{route.name}</p>
						</div>

						<div class="runner">
							{#if route.runner === 'llama.cpp'}
								<span class="lockup">{LOCKUP}llama<span class="cpp">.cpp</span></span>
							{:else}
								<img alt="" class="mark" src={hfMark} />

								<span class="runner-name">{route.runner}</span>
							{/if}
						</div>
					</li>
				{/each}
			</ol>
		</div>
	</div>
</PostFrame>

<style>
	.piece {
		display: grid;
		flex: 1;
		grid-template-rows: auto minmax(0, 1fr);
		gap: calc(var(--u) * 1.2);
		min-height: 0;
	}
	.title {
		justify-self: center;
		font-size: calc(var(--u) * 3.2);
		line-height: 0.92;
		letter-spacing: -0.035em;
	}
	/* Pi, the lines, the models and what runs them, two units in from the margin on both sides like Julien's */
	.diagram {
		display: grid;
		grid-template-columns: auto minmax(calc(var(--u) * 3), calc(var(--u) * 5)) minmax(0, 1fr) auto;
		align-items: stretch;
		padding-inline: calc(var(--u) * 2);
	}
	.diagram p {
		margin: 0;
	}
	.from {
		position: relative;
		align-self: center;
	}
	.pi {
		display: block;
		width: calc(var(--u) * 4);
		height: calc(var(--u) * 4);
		overflow: hidden;
	}
	.pi img {
		display: block;
		width: calc(var(--u) * 4 / 0.5868);
		max-width: none;
		margin: calc(var(--u) * -4 / 0.5868 * 0.2066);
	}
	/* the name and the id hang under the mark, so the mark stays on the routes' middle */
	.caption {
		position: absolute;
		top: 100%;
		left: 50%;
		display: grid;
		gap: calc(var(--u) * 0.2);
		justify-items: center;
		margin-top: calc(var(--u) * 0.6) !important;
		translate: -50% 0;
		white-space: nowrap;
	}
	.name {
		font-size: calc(var(--u) * 0.96);
		font-weight: 700;
		letter-spacing: -0.02em;
		line-height: 1;
	}
	.id {
		color: var(--p-muted);
		font: 400 calc(var(--u) * 0.62) / 1 var(--mono);
		letter-spacing: -0.03em;
	}
	/* the lines start half a unit off the mark and end a unit before the label */
	.links {
		position: relative;
		margin-inline: calc(var(--u) * 0.5) calc(var(--u) * 1);
	}
	.links svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
		fill: none;
		stroke: var(--p-ink);
		stroke-width: calc(var(--u) * 0.07);
	}
	.links path.accent {
		stroke: var(--art-cinder);
	}
	.dot {
		position: absolute;
		width: calc(var(--u) * 0.22);
		height: calc(var(--u) * 0.22);
		border-radius: 50%;
		background: var(--p-ink);
		translate: -50% -50%;
	}
	/* one row per route, spanning the model and what runs it, so each line ends at its row's middle */
	.routes {
		display: grid;
		grid-column: 3 / 5;
		grid-template-columns: subgrid;
		grid-template-rows: repeat(var(--rows), minmax(0, 1fr));
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.routes li {
		display: grid;
		grid-column: 1 / -1;
		grid-template-columns: subgrid;
		align-items: center;
	}
	/* the label 0.6u over its headline; the headline's middle on the line's end */
	.model {
		display: grid;
		gap: calc(var(--u) * 0.6);
		margin-top: calc(var(--u) * -1.14);
	}
	.t-label {
		display: flex;
		align-items: center;
		gap: calc(var(--u) * 0.3);
	}
	.tick {
		width: calc(var(--u) * 0.22);
		height: calc(var(--u) * 0.22);
		border-radius: 50%;
		background: var(--p-ink);
	}
	.tick.accent {
		background: var(--art-cinder);
	}
	.t-headline {
		white-space: nowrap;
	}
	.runner {
		display: grid;
		gap: calc(var(--u) * 0.5);
		justify-items: center;
		min-width: calc(var(--u) * 5);
		margin-left: calc(var(--u) * 2);
	}
	/* the lockup as BRAND.md types it: U+E002 and the name in Llama Mono at 90% of the size, so its icon matches
	   Llama's */
	.lockup {
		font: 700 calc(var(--u) * 1.3 * 0.9) / 1 var(--mono);
		white-space: nowrap;
	}
	.mark {
		display: block;
		width: calc(var(--u) * 2.8);
		height: calc(var(--u) * 2.8);
	}
	.runner-name {
		font-size: calc(var(--u) * 0.8);
		font-weight: 600;
		letter-spacing: -0.01em;
		line-height: 1;
		white-space: nowrap;
	}
	.cpp {
		color: var(--art-cinder);
	}
</style>
