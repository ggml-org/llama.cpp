<!-- The Llama stack, top to bottom: the app, the components it is built from, the packages under them, the engine -->
<script lang="ts">
	const layers = [
		{
			name: 'Llama Chat',
			role: 'The app: a local, private chat client, and an example to build from'
		},
		{ name: 'Llama UI', role: 'The components, tokens and patterns Llama Chat is made of' },
		{ name: 'Libraries and packages', role: 'What people take into their own apps and tools' },
		{ mono: true, name: 'llama.cpp', role: 'The engine: models running on your own machine' }
	];
</script>

<ol class="stack">
	{#each layers as layer, i (layer.name)}
		<li>
			<span class="i">{String(i + 1).padStart(2, '0')}</span>

			<span class={['name', { mono: layer.mono }]}>
				{layer.name}
			</span>

			<span class="role">{layer.role}</span>
		</li>
	{/each}
</ol>

<style>
	.stack {
		display: grid;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	li {
		display: grid;
		grid-template-columns: 2.5rem minmax(0, 1fr) minmax(0, 1.2fr);
		gap: 16px;
		align-items: baseline;
		padding-block: 18px;
		border-bottom: 1px solid var(--line-default);
	}

	li:first-child {
		border-top: 1px solid var(--fg-default);
	}

	li:last-child {
		border-bottom-color: var(--fg-default);
	}

	.i {
		color: var(--figure-muted);
		font: 500 12px var(--mono);
	}

	.name {
		font-size: clamp(22px, 3vw, 32px);
		font-weight: 600;
		line-height: 1.1;
	}

	.name.mono {
		font-family: var(--mono);
		font-size: clamp(20px, 2.7vw, 29px);
		font-weight: 700;
	}

	.role {
		color: var(--figure-muted);
		font-size: 15px;
	}

	@media (width < 640px) {
		li {
			grid-template-columns: 2rem 1fr;
		}

		.role {
			grid-column: 2;
		}
	}
</style>
