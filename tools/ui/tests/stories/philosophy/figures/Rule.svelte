<!--
	The line between Llama UI and the apps: the library draws and handles interaction, the apps decide what happens.
	Two sides of one card, with what belongs on each, after the Llama Suite's architecture page.
-->
<script lang="ts">
	const sides = [
		{
			items: [
				'Components and design tokens',
				'Multi-step flows, drag to reorder, keyboard navigation',
				'Validation as you type, formatting a file size',
				'Checking whether a model fits in memory'
			],
			name: 'In Llama UI',
			note: 'Draws and handles interaction'
		},
		{
			items: [
				'Talking to llama-server',
				'Storing conversations and notes',
				'The agentic loop and tool permissions',
				'Managing models and servers'
			],
			name: 'In the apps',
			note: 'Decide what happens'
		}
	];
</script>

<div class="rule card-surface rounded-card">
	{#each sides as side, i (side.name)}
		<section class:app={i === 1}>
			<h3>{side.name}</h3>

			<p class="note">{side.note}</p>

			<ul>
				{#each side.items as item (item)}<li>{item}</li>{/each}
			</ul>
		</section>
	{/each}
</div>

<style>
	.rule {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	section {
		display: grid;
		gap: 6px;
		align-content: start;
		padding: 20px 22px;
	}

	.app {
		border-left: 1px solid var(--line-default);
	}

	h3 {
		margin: 0;
		font-size: 18px;
		font-weight: 600;
	}

	.note {
		margin: 0;
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	ul {
		display: grid;
		gap: 6px;
		margin: 8px 0 0;
		padding-left: 1.1em;
		font-size: 15px;
		line-height: 1.45;
	}

	@media (max-width: 640px) {
		.rule {
			grid-template-columns: minmax(0, 1fr);
		}

		.app {
			border-top: 1px solid var(--line-default);
			border-left: 0;
		}
	}
</style>
