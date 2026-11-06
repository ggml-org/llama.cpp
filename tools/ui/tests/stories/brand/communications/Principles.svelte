<!--
	Communications' direction in words: what a piece should feel like, the three layers it can speak for, and how
	the section builds one, from the foundations up to the applications, as the design system builds the app.
-->
<script lang="ts">
	const principles = [
		[
			'Local is freedom',
			'A model on your own machine is yours: the weights, the conversations, the traces. You choose what runs, and no one can swap it or weaken it behind your back.'
		],
		[
			'AI for everyone',
			'Open models on the hardware people already have, and on servers people share with each other. Access to AI belongs to people everywhere, not to a subscription.'
		],
		[
			'People first',
			'A hopeful, humane future that people build for themselves. Shown, not preached: no pride, no fear, no hype.'
		],
		[
			'The real thing',
			'The app and the commands appear as they are, rendered from the design system, never mocked up or redrawn.'
		],
		[
			'Calm and exact',
			'Neutral grounds, big type and few words. Numbers are real and set in tabular figures; the Mono speaks for the engine.'
		],
		[
			'One accent',
			'Cinder, the .cpp of the logotype, marks one thing per piece: the chosen path, the figure that matters, a tag.'
		],
		[
			'Built on the unit',
			'Every size is a multiple of the unit, a 24th of the short side, so one layout works from a story to a billboard.'
		]
	];

	// how a piece speaks for each of the three layers (what they are is in Philosophy › Ideology)
	const stack = [
		['Model', 'Model cards, sizes, quants; the maker’s name, never their logo'],
		[
			'Service',
			'llama.cpp in Llama Mono, commands and figures, on the neutrals; a server can be run by one and shared by many'
		],
		['Application', 'Llama in Llama Sans, the real app, on any ground']
	];

	// the voice, by example: an early ad for llama.app, and the same thing said plainly (Applications › Open models poster)
	const voice = {
		after: 'Over 100,000 open models, running on your own computer. No account. No meter running.',
		before:
			'Download llama.app and run 100,000+ open AI models locally on any device, for free. Or would you rather keep paying for burning tokens in the cloud? Choose wisely ;)',
		why: 'The fact stays, the sneer goes. Freedom is shown by what the app does, not by mocking the alternative, and one phrase carries the accent.'
	};

	const layers = [
		['Direction', 'The mood and these principles'],
		['Foundations', 'Formats and the grid, grounds, type'],
		['Elements', 'Lockups, chips, figures, commands, motifs'],
		['Modules', 'Headline blocks, model and stat cards, the product'],
		['Key visuals', 'Layouts: Statement, Product, Roster, Bulletin, Numeral'],
		['Applications', 'Key visuals with real copy: posts, link previews, swag']
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
			Model, service and application, as Philosophy › Ideology sets them out. A piece is about one
			of them, and sounds like it.
		</p>

		<ol class="stack">
			{#each stack as [name, voice], i (name)}
				<li><span class="n">{i + 1}</span><b>{name}</b><span class="voice">{voice}</span></li>
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
