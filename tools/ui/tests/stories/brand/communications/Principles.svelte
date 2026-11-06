<!--
	Communications' direction in words: what a piece should say and how it looks, after Philosophy (Ideology, Product
	and Design); the three layers it can speak for; and how the section builds one, from the foundations up to the
	applications, as the design system builds the app.
-->
<script lang="ts">
	const principles = [
		[
			'Personal and private',
			'Show the model on the reader’s own phone, laptop or desktop. Conversations stay there, it works offline and there is no account. Say what that gives people; point at no one.'
		],
		[
			'Open to everyone',
			'Open models on hardware people already have, and on servers people share. Real machines and real sizes, not a subscription.'
		],
		[
			'Simple first',
			'Lead with what people do: a letter, a photo, a question. GGUF, quants and flags belong to pieces about the engine.'
		],
		[
			'The real thing',
			'The app and the commands appear as they are, rendered from the design system, never mocked up or redrawn.'
		],
		[
			'Plain and exact',
			'Say what the product does, with real numbers: no hype, no fear, no superlatives. Neutral grounds, big type and few words; figures in tabular Llama Mono.'
		],
		[
			'Four colours, one accent',
			'Beyond the greys, sand, the day sky and the night sky are grounds; cinder marks one thing per piece. No red.'
		],
		[
			'Play that teaches',
			'The mark can carry an idea, as the quantized L does, when the idea is true to llama.cpp and the outline stays exact.'
		],
		[
			'Built on the unit',
			'Every size is a multiple of the unit, a 24th of the short side, so one layout works from a story to a billboard.'
		]
	];

	// how a piece speaks for each of the three layers, top to bottom (what they are is in Philosophy › Ideology)
	const stack = [
		[
			'Application',
			'Llama in Llama Sans and the real app, with the harness at work in it: the loop and its tools. On any ground.'
		],
		[
			'Inference',
			'llama.cpp in Llama Mono, commands and figures, on light and dark: your own hardware, or a server someone shares'
		],
		['Model', 'Model cards, sizes, quants; the maker’s name, never their logo']
	];

	// the voice, by example: an early ad for llama.app, and the same thing said plainly (Applications › Open models poster, on the Receipt key visual)
	const voice = {
		after:
			'100,000+ open models. One computer: yours. And a receipt beside it: 1,284,902 tokens generated, no account, total $0.00.',
		before:
			'Download llama.app and run 100,000+ open AI models locally on any device, for free. Or would you rather keep paying for burning tokens in the cloud? Choose wisely ;)',
		why: 'The fact stays, the sneer goes. The receipt makes the point about paying by the token without pointing at anyone, the copy is in Llama Sans and the figures in Llama Mono, and the $0.00 carries the one accent.'
	};

	const layers = [
		['Direction', 'The mood and these principles'],
		['Foundations', 'Formats and the grid, grounds, type'],
		['Elements', 'Lockups, chips, figures, commands, motifs'],
		['Modules', 'Headline blocks, model and stat cards, the product'],
		['Key visuals', 'Layouts: Statement, Product, Roster, Bulletin, Numeral'],
		['Applications', 'Key visuals with real copy: posts, link previews, store pages, swag']
	];
</script>

<div class="page">
	<section>
		<h2>Principles</h2>

		<ol class="principles">
			{#each principles as [title, body] (title)}
				<li><b>{title}</b><span>{body}</span></li>
			{/each}
		</ol>
	</section>

	<section>
		<h2>Speaking for the three layers</h2>

		<p class="note">
			Application, inference and model, as Philosophy › Ideology sets them out, with the harness
			bridging the first two. A piece is about one of them, and sounds like it.
		</p>

		<ol class="stack">
			{#each stack as [name, voice], i (name)}
				<li>
					<span class="n">{stack.length - i}</span><b>{name}</b><span class="voice">{voice}</span>
				</li>
			{/each}
		</ol>
	</section>

	<section>
		<h2>The voice, by example</h2>

		<div class="voice">
			<p class="before"><span class="tag">Before</span>{voice.before}</p>

			<p class="after"><span class="tag">After</span>{voice.after}</p>
		</div>

		<p class="note">{voice.why}</p>
	</section>

	<section>
		<h2>From atoms to pieces</h2>

		<ol class="layers">
			{#each layers as [name, body], i (name)}
				<li><span class="n">{i}</span><b>{name}</b><span class="body">{body}</span></li>
			{/each}
		</ol>
	</section>
</div>

<style>
	.page {
		display: grid;
		gap: 3rem;
		max-width: 64rem;
	}
	h2 {
		margin: 0 0 1rem;
		font-size: 0.75rem;
		font-weight: 500;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted-foreground);
	}
	ol {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.principles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr));
		gap: 1.5rem 2rem;
	}
	.principles li {
		display: grid;
		gap: 0.4rem;
	}
	.principles b {
		font-size: 1.25rem;
		font-weight: 600;
		letter-spacing: -0.01em;
	}
	.principles span {
		color: var(--muted-foreground);
		line-height: 1.5;
	}
	.stack {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(16rem, 1fr));
		gap: 1rem;
	}
	.stack li {
		display: grid;
		grid-template-columns: auto 1fr;
		align-content: start;
		gap: 0.4rem 0.75rem;
		padding: 1rem 1.25rem;
		border: 1px solid var(--border);
		border-radius: 0.75rem;
	}
	.stack b {
		font-size: 1.25rem;
		font-weight: 600;
	}
	.note {
		margin: 0 0 1rem;
		color: var(--muted-foreground);
	}
	.stack .voice {
		grid-column: 1 / -1;
		line-height: 1.5;
	}
	.voice {
		color: var(--muted-foreground);
	}
	.voice {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr));
		gap: 1rem;
		margin-bottom: 1rem;
	}
	.voice p {
		display: grid;
		gap: 0.5rem;
		align-content: start;
		margin: 0;
		padding: 1rem 1.25rem;
		border: 1px solid var(--border);
		border-radius: 0.75rem;
		font: 400 0.95rem / 1.5 var(--font-mono);
	}
	.before {
		color: var(--muted-foreground);
	}
	.tag {
		color: var(--muted-foreground);
		font: 500 0.7rem var(--font-mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
	.layers li {
		display: grid;
		grid-template-columns: 2rem 9rem 1fr;
		align-items: baseline;
		gap: 1rem;
		padding: 0.75rem 0;
		border-top: 1px solid var(--border);
	}
	.n {
		color: var(--art-cinder);
		font-family: var(--font-mono);
	}
	.layers b {
		font-weight: 600;
	}
	.body {
		color: var(--muted-foreground);
	}
	@media (max-width: 560px) {
		.layers li {
			grid-template-columns: 2rem 1fr;
		}
		.body {
			grid-column: 2;
		}
	}
</style>
