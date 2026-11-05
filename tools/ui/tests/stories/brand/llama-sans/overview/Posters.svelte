<!--
	Llama Sans Display at poster scale: four pieces in fixed palettes (they read the same in both themes), after the
	GitHub poster series: words on a dot grid, a stacked statement, a big word over a branch graph and a word stack.
-->
<script lang="ts">
	// the dot-grid poster: its rows, words among dots ('·'), then rows of dots alone
	const rows = [
		['Run', '·', '·', '·', '·', '·'],
		['·', 'the', '·', 'model'],
		['on', '·', 'your', '·', '·'],
		['·', '·', '·', 'own', '·'],
		['·', 'machine', '·', '·'],
		['·', '·', '·', '·', '·', '·', '·', '·'],
		['·', '·', '·', '·', '·', '·', '·', '·']
	];

	// the branch graph under GGUF: each branch forks down from the trunk at `from`, runs along `y` past its commits
	// (`nodes`) and merges back at `to`
	const branches = [
		{ from: 40, nodes: [90], to: 140, y: 50 },
		{ from: 120, nodes: [175, 230], to: 290, y: 90 },
		{ from: 200, nodes: [255], to: 330, y: 130 }
	];
	const R = 12; // the corners where a branch turns
</script>

<div class="posters">
	<article aria-label="Run the model on your own machine" class="poster dots">
		<div class="field">
			{#each rows as row, r (r)}
				<p class="row">
					{#each row as token, i (i)}{#if token === '·'}<i></i>{:else}<b>{token}</b>{/if}{/each}
				</p>
			{/each}
		</div>

		<footer><span>Make it local</span><span class="icon"></span></footer>
	</article>

	<article aria-label="If it fits in memory, it runs" class="poster stack">
		<p class="top">If it fits<br />in memory</p>

		<p class="mark"></p>

		<p class="bottom">it runs.</p>

		<footer><span>llama.cpp</span><span>Local inference, any hardware</span></footer>
	</article>

	<article aria-label="GGUF" class="poster gguf">
		<p class="word">GGUF</p>

		<svg aria-hidden="true" viewBox="0 0 360 150">
			<line x1="10" x2="350" y1="12" y2="12" />

			<circle cx="10" cy="12" r="7" />

			<circle cx="350" cy="12" r="7" />

			{#each branches as b (b.y)}
				<path
					d="M{b.from} 12 V{b.y - R} Q{b.from} {b.y} {b.from + R} {b.y} H{b.to -
						R} Q{b.to} {b.y} {b.to} {b.y - R} V12"
				/>

				{#each b.nodes as x (x)}<circle cx={x} cy={b.y} r="7" />{/each}
			{/each}
		</svg>

		<p class="text">
			GGUF is the file format llama.cpp loads: one file holding a model's weights, its tokenizer and
			everything needed to run it, quantised to fit the machine at hand.
		</p>

		<footer><span> llama.cpp</span></footer>
	</article>

	<article aria-label="Local, not remote" class="poster words">
		<p class="w1">local</p>

		<p class="aside a1">Your prompts stay on your machine, and so do the answers.</p>

		<p class="w2">not</p>

		<p class="w3">remote</p>

		<p class="aside a2">At ggml we build for the hardware you already have.</p>
	</article>
</div>

<style>
	.posters {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 16px;
		width: var(--column);
		margin-inline: auto;
	}
	@media (max-width: 900px) {
		.posters {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	.poster {
		position: relative;
		display: grid;
		container-type: inline-size;
		overflow: hidden;
		box-sizing: border-box;
		aspect-ratio: 3 / 4;
		padding: 7%; /* of the card's width; the type inside scales with the card (cqi) */
		border-radius: 6px;
		font-family: var(--headline);
	}
	.poster p {
		margin: 0;
	}
	.poster footer {
		display: flex;
		justify-content: space-between;
		align-items: end;
		font: 400 3.6cqi / 1.25 var(--ui);
		opacity: 0.6;
	}

	/* words placed on a dot grid, dark grey */
	.dots {
		grid-template-rows: 1fr auto;
		background: var(--ds-neutral-900);
		color: var(--ds-neutral-50);
	}
	.field {
		display: grid;
		gap: 3.4cqi;
		align-content: start;
	}
	.row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		height: 12cqi;
	}
	.row b {
		font: 700 12.5cqi / 1 var(--headline);
		letter-spacing: -0.01em;
	}
	.row i {
		width: 1.6cqi;
		height: 1.6cqi;
		border-radius: 50%;
		background: rgb(255 255 255 / 0.22);
	}
	.dots .icon {
		font: 700 7cqi / 1 var(--headline);
		color: var(--beige);
		opacity: 1;
	}

	/* a stacked statement, beige */
	.stack {
		grid-template-rows: auto 1fr auto auto;
		background: var(--beige);
		color: var(--beige-ink);
	}
	.stack .top,
	.stack .bottom {
		font: 700 16cqi / 0.92 var(--headline);
		letter-spacing: -0.02em;
		text-transform: uppercase;
	}
	.stack .bottom {
		text-align: right;
	}
	.stack .mark {
		align-self: center;
		justify-self: center;
		font: 700 34cqi / 1 var(--headline);
		color: var(--clay);
	}
	.stack footer {
		margin-top: 5cqi;
	}

	/* a big word over a branch graph, on llama.app's soft blue */
	.gguf {
		grid-template-rows: auto auto 1fr auto;
		gap: 4%; /* of the card: its own sizes can't use cqi, which measures the card's container */
		background: var(--mist);
		color: var(--mist-ink);
	}
	.gguf .word {
		font: 700 33cqi / 0.8 var(--headline);
		letter-spacing: -0.04em;
	}
	.gguf svg {
		display: block;
		width: 100%;
		height: auto;
		aspect-ratio: 360 / 150;
		overflow: visible; /* the end nodes sit on the edge */
	}
	.gguf svg :is(line, path) {
		fill: none;
		stroke: var(--mist-ink);
		stroke-width: 2.5;
	}
	.gguf svg circle {
		fill: #d6e7f8;
		stroke: var(--mist-ink);
		stroke-width: 2.5;
	}
	.gguf .text {
		align-self: end;
		max-width: 34ch;
		font: 500 3.4cqi / 1.4 var(--ui);
	}
	.gguf footer {
		justify-content: end;
		font-weight: 700;
		opacity: 1;
	}

	/* a word stack on llama.app's navy */
	.words {
		grid-template-rows: repeat(4, auto);
		align-content: space-between;
		background: var(--llama-navy);
		color: #fff;
	}
	.words :is(.w1, .w2, .w3) {
		font: 700 27cqi / 0.86 var(--headline);
		letter-spacing: -0.04em;
	}
	.words .w2 {
		text-align: right;
	}
	.words .aside {
		max-width: 42cqi;
		font: 600 3.4cqi / 1.3 var(--ui);
	}
	.words .a2 {
		justify-self: end;
		text-align: right;
	}
</style>
