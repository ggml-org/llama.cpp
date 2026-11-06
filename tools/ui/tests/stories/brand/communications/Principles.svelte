<!--
	Communications' direction in words: what a piece should feel like, and how the section builds one, from the
	foundations up to the applications, as the design system builds the app.
-->
<script lang="ts">
	const principles = [
		[
			'Local, and proud of it',
			'Every piece is about running AI on your own machine. Show the machine: the menu bar, a terminal, a file size that fits.'
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
		],
		[
			'Two voices',
			'Llama, the app, on the bright grounds; llama.cpp, the engine, in Llama Mono on the neutrals.'
		]
	];

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
