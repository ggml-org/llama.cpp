<!--
	The three layers between a person and the AI, top to bottom: the application people use (with its harness, the
	agentic loop and its tools), inference (the engine and server) and the model's weights. Each says what it is, what
	it is in the Llama stack, and what it means to have it on your side. The application, the one people touch, carries
	the cinder mark.
-->
<script lang="ts">
	const layers = [
		{
			here: 'Llama Chat, Llama for Mac and Windows',
			name: 'Application',
			what: 'The UI and UX people work with, and its harness: the agentic loop and the tools it calls.',
			yours: 'Your conversations kept where you are; only the tools you added'
		},
		{
			here: 'ggml, libllama and llama-server',
			name: 'Inference',
			what: 'The engine and the server that run the model.',
			yours: 'On your hardware, or a server someone you trust shares'
		},
		{
			here: 'A GGUF file, from Hugging Face',
			name: 'Model',
			what: 'The weights: what everything else is built on.',
			yours: 'Open, and the exact model you chose'
		}
	];
</script>

<ol class="layers">
	{#each layers as layer, i (layer.name)}
		<li class:top={i === 0} class="layer card-surface rounded-card">
			<span class="n">{layers.length - i}</span>

			<h3>{layer.name}</h3>

			<p class="what">{layer.what}</p>

			<p class="here"><code>{layer.here}</code></p>

			<p class="yours">{layer.yours}</p>
		</li>
	{/each}
</ol>

<style>
	.layers {
		display: grid;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.layer {
		display: grid;
		grid-template-columns: 2rem 9rem minmax(0, 1fr) minmax(0, 1fr);
		gap: 6px 16px;
		align-items: baseline;
		padding: 16px 20px;
	}

	.n {
		color: var(--figure-muted);
		font: 500 12px var(--mono);
	}

	/* the one accent: the layer people touch */
	.top .n {
		color: var(--art-cinder);
	}

	h3 {
		margin: 0;
		font-size: 18px;
		font-weight: 600;
	}

	p {
		margin: 0;
		font-size: 15px;
		line-height: 1.45;
	}

	.what {
		grid-column: 3;
	}

	.here {
		grid-column: 3;
		color: var(--figure-muted);
	}

	.here code {
		font: 500 13px var(--mono);
	}

	.yours {
		grid-column: 4;
		grid-row: 1 / span 2;
		color: var(--figure-muted);
	}

	@media (max-width: 640px) {
		.layer {
			grid-template-columns: 2rem minmax(0, 1fr);
		}

		.what,
		.here,
		.yours {
			grid-column: 2;
			grid-row: auto;
		}
	}
</style>
