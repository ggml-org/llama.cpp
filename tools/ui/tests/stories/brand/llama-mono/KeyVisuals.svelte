<!--
	Llama Mono's key visuals: four neutral posters in fixed greys (they read the same in both themes), each set in the
	font's own cells. A terminal starting llama-server, the look-alikes kept apart, a generation speed in tabular
	figures, and the first bytes of a GGUF file as a hex dump. The cinder cursor is the one accent.
-->
<script lang="ts">
	const terminal = [
		{ kind: 'prompt', text: 'llama-server -hf ggml-org/gemma-3-1b-it-GGUF' },
		{ kind: 'log', text: 'load_model: loading model' },
		{ kind: 'log', text: 'offloaded 27/27 layers to GPU' },
		{ kind: 'log', text: 'main: model loaded' },
		{ kind: 'ok', text: 'server is listening on' },
		{ kind: 'ok', text: 'http://127.0.0.1:9931' }
	];

	// "GGUF", version 3, then the tensor and key-value counts (little-endian u64s)
	const hex = [
		['47 47 55 46', '03 00 00 00', 'GGUF....'],
		['22 01 00 00', '00 00 00 00', '".......'],
		['1d 00 00 00', '00 00 00 00', '........'],
		['14 00 00 00', '00 00 00 00', '........'],
		['67 65 6e 65', '72 61 6c 2e', 'general.'],
		['61 72 63 68', '69 74 65 63', 'architec'],
		['74 75 72 65', '08 00 00 00', 'ture....']
	];

	const bars = [0.42, 0.58, 0.66, 0.71, 0.74, 0.77, 0.8, 0.79, 0.82, 0.84, 0.83, 0.86];
</script>

<div class="visuals">
	<article aria-label="A terminal starting llama-server" class="poster term">
		<div class="lines">
			{#each terminal as l, i (i)}
				<p class={l.kind}>
					{#if l.kind === 'prompt'}<b>$</b>{/if}{l.text}
				</p>
			{/each}

			<p class="prompt"><b>$</b><i class="cursor"></i></p>
		</div>

		<footer><span>Llama Mono</span><span class="icon"></span></footer>
	</article>

	<article aria-label="Look-alikes kept apart" class="poster apart">
		<p class="big">0O</p>

		<p class="big">1lI|</p>

		<p class="small">Every one of them keeps its own shape, and every one the same cell.</p>

		<footer><span>Kept apart</span><span>600 units</span></footer>
	</article>

	<article aria-label="86.4 tokens a second" class="poster speed">
		<p class="label">eval</p>

		<p class="number">86.4</p>

		<p class="unit">tokens / s</p>

		<div aria-hidden="true" class="bars">
			{#each bars as b, i (i)}<span style:height="{b * 100}%"></span>{/each}
		</div>

		<dl>
			<dt>prompt</dt>

			<dd>1 024 t · 0.31 s</dd>

			<dt>eval</dt>

			<dd>&nbsp;&nbsp;512 t · 5.93 s</dd>
		</dl>
	</article>

	<article aria-label="The first bytes of a GGUF file" class="poster dump">
		<p class="title">model.gguf</p>

		<div class="rows">
			{#each hex as [a, b, ascii], i (i)}
				<p>
					<span class="offset">{(i * 8).toString(16).padStart(4, '0')}</span>

					<span>{a}</span>

					<span>{b}</span>

					<span class="ascii">{ascii}</span>
				</p>
			{/each}
		</div>

		<footer><span>One file, the whole model</span><span>llama.cpp</span></footer>
	</article>
</div>

<style>
	.visuals {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 16px;
		width: var(--column);
		margin-inline: auto;
	}
	@media (max-width: 900px) {
		.visuals {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	.poster {
		display: grid;
		container-type: inline-size;
		overflow: hidden;
		box-sizing: border-box;
		aspect-ratio: 3 / 4;
		padding: 7%; /* of the card's width; the type inside scales with the card (cqi) */
		border-radius: 6px;
		font-family: var(--code);
	}
	.poster p,
	.poster dl,
	.poster dd {
		margin: 0;
	}
	.poster footer {
		display: flex;
		justify-content: space-between;
		align-items: end;
		font: 400 3.6cqi / 1.25 var(--code);
		opacity: 0.6;
	}

	/* a terminal, near black */
	.term {
		grid-template-rows: 1fr auto;
		background: var(--ds-neutral-950);
		color: var(--ds-neutral-50);
	}
	.lines {
		display: grid;
		gap: 2.2cqi;
		align-content: start;
		font: 400 4.6cqi / 1.35 var(--code);
	}
	.lines b {
		margin-right: 1ch;
		font-weight: 700;
		color: var(--ds-neutral-400);
	}
	.lines .prompt {
		font-weight: 600;
	}
	.lines .log {
		color: var(--ds-neutral-400);
	}
	.lines .ok {
		font-weight: 500;
	}
	.cursor {
		display: inline-block;
		width: 1ch;
		height: 1.1em;
		vertical-align: -0.2em;
		background: var(--cinder);
	}
	.term .icon {
		font: 700 6cqi / 1 var(--code);
		color: var(--ds-neutral-400);
		opacity: 1;
	}

	/* look-alikes at poster size, light grey */
	.apart {
		grid-template-rows: auto auto 1fr auto;
		background: var(--ds-neutral-100);
		color: var(--ds-neutral-950);
	}
	.apart .big {
		font: 700 30cqi / 0.95 var(--code);
		letter-spacing: -0.02em;
	}
	.apart .small {
		align-self: end;
		max-width: 80%;
		margin-bottom: 5cqi;
		font: 400 4.2cqi / 1.35 var(--ui);
	}

	/* a generation speed in tabular figures, soft grey */
	.speed {
		grid-template-rows: auto auto auto 1fr auto;
		background: linear-gradient(160deg, #e2e2e2 0%, #ececec 100%);
		color: var(--ds-neutral-950);
	}
	.speed .label,
	.speed .unit {
		font: 500 4.4cqi / 1.2 var(--code);
		text-transform: uppercase;
		letter-spacing: 0.08em;
		opacity: 0.6;
	}
	.speed .number {
		font: 700 34cqi / 0.9 var(--code);
		letter-spacing: -0.04em;
	}
	.bars {
		display: flex;
		align-items: end;
		gap: 1.6cqi;
		height: 22cqi;
		margin-block: 6cqi;
	}
	.bars span {
		flex: 1;
		border-radius: 1cqi 1cqi 0 0;
		background: var(--ds-neutral-950);
		opacity: 0.85;
	}
	.speed dl {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 1cqi 4cqi;
		font: 400 4cqi / 1.3 var(--code);
	}
	.speed dt {
		opacity: 0.6;
	}
	.speed dd {
		white-space: pre;
	}

	/* the first bytes of a GGUF file, white */
	.dump {
		grid-template-rows: auto 1fr auto;
		gap: 5cqi;
		background: var(--ds-neutral-0);
		color: var(--ds-neutral-950);
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.08);
	}
	.dump .title {
		font: 700 9cqi / 1 var(--code);
		letter-spacing: -0.02em;
	}
	.rows {
		display: grid;
		gap: 1.8cqi;
		align-content: start;
		font: 400 3.5cqi / 1.3 var(--code);
	}
	.rows p {
		display: flex;
		justify-content: space-between;
		gap: 1ch;
		white-space: pre;
	}
	.rows p:first-child {
		font-weight: 700;
	}
	.rows .offset,
	.rows .ascii {
		opacity: 0.45;
	}
	.rows p:first-child .ascii {
		opacity: 1;
		color: var(--cinder);
	}
</style>
