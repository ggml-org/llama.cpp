<!-- Llama Sans UI at work in running text: reference, data, an index, docs, a size ladder, a table and a newsletter.
     Body copy is set at real reading sizes in px, at line-height 1.5; headings and labels keep the font's own. -->
<script lang="ts">
	const steps = ['Tokenize', 'Embed', 'Attend', 'Sample', 'Decode'];
	const sizes = [12, 14, 16, 18, 20];
	const models: [string, string, string, string][] = [
		['Llama 3.2 1B', '1.24', '0.81', '212.4'],
		['Qwen 2.5 7B', '7.62', '4.68', '61.9'],
		['Mistral Nemo', '12.2', '7.48', '38.7'],
		['Gemma 2 27B', '27.2', '16.6', '17.3'],
		['Llama 3.3 70B', '70.6', '42.5', '6.8']
	];
</script>

<section aria-label="Llama Sans UI in running text" class="gallery prose">
	<!-- a reference card: a paragraph, then one big word -->
	<article class="card c-hero reference">
		<p class="para">
			<b>Llamas</b> have carried loads across the Andes for thousands of years. Social and steady-footed,
			they walk in strings of a dozen or more, each carrying about a quarter of its own weight over ground
			no cart could cross.
		</p>

		<p class="word">Llama</p>
	</article>

	<!-- a data card, after the periodic table -->
	<article class="card c-paper data">
		<div class="sym">Q4</div>

		<dl class="facts">
			<dt>4-bit</dt>

			<dd class="name">Q4_K_M<br /><span>4.5 bits per weight</span></dd>

			<dd class="foot">(4.07 GB, 7B)</dd>
		</dl>
	</article>

	<!-- an index, alternating weights on fine rules -->
	<article class="card c-ink index">
		<ol>
			{#each steps as s, i (s)}<li class:strong={i % 2 === 0}>
					<span>{String(i + 1).padStart(2, '0')}</span>{s}
				</li>{/each}
		</ol>
	</article>

	<!-- a documentation page -->
	<article class="card wide free c-paper docs">
		<nav aria-label="Docs">
			<p class="mono-label">llama.cpp docs</p>

			<ul>
				<li>Getting started</li>

				<li class="on">Run a model</li>

				<li>Build from source</li>

				<li>Quantize</li>

				<li>Server &amp; API</li>
			</ul>
		</nav>

		<div class="page">
			<h3>Run a model</h3>

			<p>
				Download a model in GGUF format, then point <code>llama-cli</code> at it. The first run maps the
				weights into memory, so later runs start in under a second.
			</p>

			<pre><code>llama-cli -m models/qwen2.5-7b-q4_k_m.gguf -p "Hello"</code></pre>

			<p class="note">
				<b>Tip.</b> Add <code>-ngl 99</code> to offload every layer to the GPU when it fits in memory.
			</p>

			<h4>Next</h4>

			<ul class="next">
				<li>Serve the model over an OpenAI-compatible API with <code>llama-server</code>.</li>

				<li>Shrink it further with <code>llama-quantize</code>, from Q8_0 down to Q2_K.</li>
			</ul>
		</div>
	</article>

	<!-- the same sentence down the reading sizes -->
	<article class="card free c-paper ladder">
		<p class="mono-label">Reading sizes</p>

		{#each sizes as px (px)}
			<p class="row">
				<span>{px}</span><span style:font-size="{px}px"
					>Steady letters keep long reading easy on the eye.</span
				>
			</p>
		{/each}
	</article>

	<!-- a table of tabular figures -->
	<article class="card free c-paper table">
		<p class="mono-label">Sample figures · Q4_K_M · tokens per second</p>

		<table>
			<thead><tr><th>Model</th><th>Params (B)</th><th>Size (GB)</th><th>Tok/s</th></tr></thead>

			<tbody>
				{#each models as [m, p, s, r] (m)}<tr><td>{m}</td><td>{p}</td><td>{s}</td><td>{r}</td></tr
					>{/each}
			</tbody>
		</table>
	</article>

	<!-- a newsletter -->
	<article class="card wide free c-ink letter">
		<p class="mono-label">The llama.cpp letter · No. 12</p>

		<h3>This month in local inference</h3>

		<p>Hi everyone,</p>

		<p>
			Three things landed since the last letter: faster prompt processing on Apple silicon, a new
			server dashboard, and sampling presets you can share as a single file. Thank you to the 41
			people who sent patches.
		</p>

		<p class="sign">— The maintainers</p>
	</article>
</section>

<style>
	.prose {
		font-family: var(--display);
	}

	/* reference */
	.reference {
		display: grid;
		align-content: space-between;
		padding: 22px 24px;
	}
	.reference .para {
		padding-bottom: 14px;
		border-bottom: 1px solid var(--llama-hero-rule);
		font: 400 15px / 1.5 var(--display);
	}
	.reference .para b {
		font-weight: 700;
	}
	.reference .word {
		padding-top: 6px;
		border-top: 1px solid var(--llama-hero-rule);
		font: 700 24cqi / 0.95 var(--display);
		letter-spacing: -0.01em;
	}

	/* data */
	.data {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 3cqi;
		padding: 3cqi;
		background: var(--ds-neutral-200);
	}
	.data .sym {
		display: grid;
		place-items: center;
		border: 1px solid rgb(0 0 0 / 0.12);
		border-radius: 1cqi;
		font: 700 22cqi / 1 var(--display);
		color: var(--brand-ink);
	}
	.data .facts {
		display: grid;
		grid-template-rows: auto 1fr auto;
		padding: 4cqi;
		border-radius: 1cqi;
		background: #fff;
		color: var(--brand-ink);
	}
	.data dt {
		font: 700 18px / normal var(--display);
	}
	.data .name {
		align-self: center;
		text-align: center;
		font: 500 18px / normal var(--display);
	}
	.data .name span {
		font-size: 14px;
		font-weight: 400;
	}
	.data .foot {
		text-align: center;
		font: 400 15px / normal var(--display);
		font-variant-numeric: tabular-nums;
	}

	/* index */
	.index {
		display: grid;
		align-content: center;
		padding: 5cqi 6cqi;
	}
	.index ol {
		padding: 0;
		list-style: none;
	}
	.index li {
		display: flex;
		gap: 4cqi;
		align-items: baseline;
		padding-block: 1.4cqi;
		border-top: 1px solid rgb(255 255 255 / 0.22);
		font: 400 8cqi / 1.1 var(--display);
	}
	.index li:last-child {
		border-bottom: 1px solid rgb(255 255 255 / 0.22);
	}
	.index li.strong {
		font-weight: 700;
	}
	.index li span {
		width: 9cqi;
		font-weight: 400;
		font-variant-numeric: tabular-nums;
		opacity: 0.7;
	}

	/* docs */
	.docs {
		display: grid;
		grid-template-columns: 180px 1fr;
		gap: 28px;
		padding: 22px 26px;
		background: var(--ds-neutral-50);
	}
	.docs nav {
		display: grid;
		gap: 12px;
		align-content: start;
		padding-right: 22px;
		border-right: 1px solid rgb(0 0 0 / 0.1);
	}
	.docs nav ul {
		display: grid;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
		font: 400 14px / normal var(--display);
	}
	.docs nav li {
		padding: 3px 8px;
		border-radius: 8px;
		color: var(--ds-neutral-500);
	}
	.docs nav li.on {
		background: var(--brand-ink);
		color: #fff;
		font-weight: 600;
	}
	.docs .page {
		display: grid;
		gap: 12px;
		align-content: start;
		max-width: 62ch;
		font: 400 15px / 1.5 var(--display);
	}
	.docs h3 {
		font: 700 24px / normal var(--display);
	}
	.docs code {
		padding: 1px 5px;
		border-radius: 4px;
		background: var(--ds-neutral-150);
		font: 400 0.9em var(--mono);
	}
	.docs pre {
		margin: 0;
		padding: 12px 14px;
		overflow-x: auto;
		border-radius: 8px;
		background: var(--brand-ink);
		color: var(--brand-light);
	}
	.docs pre code {
		padding: 0;
		background: none;
		font-size: 13px;
	}
	.docs h4 {
		margin: 6px 0 0;
		font: 700 17px / normal var(--display);
	}
	.docs .next {
		display: grid;
		gap: 4px;
		margin: 0;
		padding-left: 18px;
	}
	.docs .note {
		padding: 10px 14px;
		border-left: 3px solid var(--clay);
		border-radius: 0 8px 8px 0;
		background: color-mix(in srgb, var(--clay) 12%, white);
		font-size: 14px;
	}
	@media (max-width: 700px) {
		.docs {
			grid-template-columns: 1fr;
			gap: 16px;
		}
		.docs nav {
			padding: 0 0 14px;
			border-right: 0;
			border-bottom: 1px solid rgb(0 0 0 / 0.1);
		}
		.docs nav ul {
			display: flex;
			flex-wrap: wrap;
		}
	}

	/* size ladder */
	.ladder {
		display: grid;
		gap: 10px;
		align-content: start;
		padding: 22px 24px;
	}
	.ladder .row {
		display: grid;
		grid-template-columns: 26px 1fr;
		gap: 10px;
		align-items: baseline;
		padding-top: 8px;
		border-top: 1px solid rgb(0 0 0 / 0.1);
		font-family: var(--display);
		line-height: 1.5;
	}
	.ladder .row span:first-child {
		font: 500 11px var(--mono);
		opacity: 0.6;
	}

	/* table */
	.table {
		display: grid;
		gap: 14px;
		align-content: start;
		padding: 22px 24px;
	}
	.table table {
		width: 100%;
		border-collapse: collapse;
		font: 400 15px / normal var(--display);
		font-variant-numeric: tabular-nums;
	}
	.table th {
		padding: 0 0 8px;
		border-bottom: 2px solid var(--brand-ink);
		font-weight: 600;
		font-size: 13px;
		text-align: right;
	}
	.table td {
		padding: 7px 0;
		border-bottom: 1px solid rgb(0 0 0 / 0.14);
		text-align: right;
	}
	.table th:first-child,
	.table td:first-child {
		text-align: left;
	}
	.table td:last-child {
		font-weight: 700;
	}

	/* newsletter */
	.letter {
		display: grid;
		gap: 12px;
		align-content: start;
		padding: 22px 24px;
		font: 400 15px / 1.5 var(--display);
	}
	.letter h3 {
		font: 700 22px / normal var(--display);
		color: #fff;
	}
	.letter p {
		max-width: 62ch;
	}
	.letter .sign {
		opacity: 0.7;
	}
</style>
