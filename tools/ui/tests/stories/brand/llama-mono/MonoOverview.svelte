<!--
	Llama Mono at a glance, in the manner of the Llama Sans overview: the name at full size, what it is, code in it, the
	characters that must stay apart, the weights, the character set and the key visuals. It sits in OverviewFrame,
	which brings the specimen's colours; Llama Mono itself is --code.
-->
<script lang="ts">
	import Section from '../llama-sans/overview/Section.svelte';
	import KeyVisuals from './KeyVisuals.svelte';

	const weights = [
		{ name: 'Regular', weight: 400 },
		{ name: 'Medium', weight: 500 },
		{ name: 'SemiBold', weight: 600 },
		{ name: 'Bold', weight: 700 }
	];

	// the look-alikes code has to keep apart, in pairs and threes
	const distinctions = ['0 O o', '1 l I |', 'rn m', ', . ;', '\' ` "', '- – —', '{ ( [', '5 S s'];

	const ligatures = ['->', '=>', '<=', '>=', '!=', '==', '===', '!==', '<=>', '|>', '::', '...'];

	const code = `// load a GGUF model and print what it holds
int main(int argc, char ** argv) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;            // offload everything that fits

    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) {
        fprintf(stderr, "error: unable to load %s\\n", argv[1]);
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("vocab: %d tokens, 0x%04X\\n", n_vocab, 0x1F40);  // ilI1| O0 -> =>
    llama_model_free(model);
}`;

	const charset = [
		'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
		'abcdefghijklmnopqrstuvwxyz',
		'0123456789 ½ ¼ ¾ ² ³',
		'!?.,:;…·•‘’“”«»‹›',
		'()[]{}<>/\\|¦_-–—',
		'+−×÷=≠≈<>≤≥±~^%‰#&@$€£¥¢',
		'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØŒÙÚÛÜÝ',
		'àáâãäåæçèéêëìíîïñòóôõöøœùúûüýÿß'
	];
</script>

<header class="hero">
	<p class="kicker">The monospace of the Llama family</p>

	<p aria-hidden="true" class="icon"></p>

	<h1>Llama<br />Mono</h1>

	<p class="tagline">
		<span>Every character, the same cell.</span><span>Llama Sans's letters, set for code.</span>
	</p>
</header>

<Section columns label="For code and numbers">
	<p>
		Llama Mono is Llama Sans set in a fixed cell: built from Geist Mono with the same weights, the
		tailless a, the narrow t, the angle-cut terminals, the rounder curves and the deeper stem joins.
		Every glyph keeps a 600-unit cell, so columns line up in a terminal, a diff or a table of
		numbers.
	</p>

	<p>
		Where a Llama Sans change would make a letter wider or narrower, the drawing moves back to the
		centre of its cell instead. The l keeps its serifs and G and S keep their width, so the letters
		code needs to tell apart stay apart. The Llama icon sits in the font at U+E000, sized to the
		cell.
	</p>
</Section>

<Section label="In code">
	<pre class="code">{code}</pre>
</Section>

<Section label="Kept apart">
	<div class="pairs">
		{#each distinctions as d (d)}<span>{d}</span>{/each}
	</div>
</Section>

<Section label="Ligatures">
	<div class="ligatures">
		{#each ligatures as l (l)}<span>{l}</span>{/each}
	</div>

	<p class="note">
		Each ligature keeps one cell per character, so the columns after it stay in line.
	</p>
</Section>

<Section label="Weights">
	<div class="weights">
		{#each weights as w (w.weight)}
			<p style:font-weight={w.weight}>
				<span class="label">{w.name} {w.weight}</span>

				<span class="line">llama-server --port 8080</span>
			</p>
		{/each}
	</div>
</Section>

<Section label="Characters">
	<div class="charset">
		{#each charset as row (row)}<p>{row}</p>{/each}
	</div>
</Section>

<KeyVisuals />

<style>
	.hero {
		display: grid;
		justify-items: center;
		gap: clamp(20px, 3vw, 36px);
		padding: clamp(56px, 9vw, 120px) 16px clamp(64px, 10vw, 136px);
		background: linear-gradient(160deg, var(--sheet) 0%, var(--paper) 100%);
		color: var(--ink);
		text-align: center;
	}
	.hero > * {
		margin: 0;
	}
	.kicker {
		font: 500 12px var(--code);
		letter-spacing: 0.12em;
		text-transform: uppercase;
		opacity: 0.7;
	}
	.icon {
		font: 700 clamp(40px, 6vw, 72px) / 1 var(--code);
		opacity: 0.85;
	}
	h1 {
		font: 700 clamp(72px, 15vw, 208px) / 0.9 var(--code);
		letter-spacing: -0.02em;
	}
	.tagline {
		display: grid;
		gap: 2px;
		font: 400 clamp(18px, 2.4vw, 26px) / 1.3 var(--ui);
		opacity: 0.85;
	}

	.code {
		margin: 0;
		padding: 20px 24px;
		overflow-x: auto;
		border-radius: 12px;
		background: var(--sheet);
		font: 400 clamp(12px, 1.5vw, 16px) / 1.6 var(--code);
	}

	.pairs {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		gap: 8px;
	}
	.pairs span {
		padding: 18px 12px;
		border-radius: 10px;
		background: var(--sheet);
		font: 500 clamp(24px, 3vw, 34px) / 1 var(--code);
		text-align: center;
		white-space: pre;
	}

	.ligatures {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
		gap: 8px;
	}
	.ligatures span {
		padding: 18px 12px;
		border-radius: 10px;
		background: var(--sheet);
		font: 500 clamp(24px, 3vw, 34px) / 1 var(--code);
		text-align: center;
		white-space: pre;
	}
	.note {
		font: 400 14px / 1.5 var(--ui);
		opacity: 0.7;
	}

	.weights {
		display: grid;
		gap: 4px;
	}
	.weights p {
		display: grid;
		grid-template-columns: 9rem 1fr;
		align-items: baseline;
		gap: 16px;
		margin: 0;
		font-family: var(--code);
	}
	.weights .label {
		font: 400 12px var(--ui);
		opacity: 0.6;
	}
	.weights .line {
		font-size: clamp(20px, 3.4vw, 40px);
		line-height: 1.3;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.charset p {
		margin: 0;
		font: 400 clamp(16px, 2.2vw, 26px) / 1.5 var(--code);
		overflow-wrap: anywhere;
	}
</style>
