<!--
	The Llama stack across its three layers, top to bottom: the application people use, with its harness (the agentic
	loop and the tools it calls) inside it, inference that runs the model, and the model's open weights, each with what
	it is for and its real parts. Code names (ggml, libllama, GGUF, MCP) are set in Llama Mono. The slot for apps that
	are not ours is dashed, as in the Introduction's building blocks; the application layer, the one people touch,
	carries the cinder mark.
-->
<script lang="ts">
	interface Part {
		/** a slot for someone else's work, dashed */
		open?: boolean;
		code?: boolean;
		name: string;
		note: string;
	}

	const layers: { harness?: Part[]; name: string; parts: Part[]; who: string }[] = [
		{
			harness: [
				{ name: 'Agentic loop', note: 'In Llama Chat: calls tools and feeds back their results' },
				{ name: 'Server tools', note: 'Run by llama-server, next to the model' },
				{ name: 'Browser tools', note: 'Run in the page, on your device' },
				{ code: true, name: 'MCP', note: 'Tools and resources from the servers you add' }
			],
			name: 'Application',
			parts: [
				{ name: 'Llama Chat', note: 'The chat client, bundled with llama-server' },
				{ name: 'Llama for Mac and Windows', note: 'Run the server from the menu bar or tray' },
				{ name: 'Your app', note: 'On the server’s API, with Llama UI or without', open: true }
			],
			who: 'What people use'
		},
		{
			name: 'Inference',
			parts: [
				{ code: true, name: 'llama-server', note: 'HTTP, with an OpenAI-compatible API' },
				{ code: true, name: 'libllama', note: 'Loads a GGUF and runs it' },
				{ code: true, name: 'ggml', note: 'The tensor library: CPU, Metal, CUDA, Vulkan and more' },
				{ name: 'Your hardware', note: 'A phone, a laptop, a desktop or a server' }
			],
			who: 'What runs it'
		},
		{
			name: 'Model',
			parts: [
				{ code: true, name: 'GGUF', note: 'One file: weights, tokenizer, chat template' },
				{
					code: true,
					name: 'Q8_0 … Q2_K',
					note: 'Quantizations by llama-quantize, to fit your memory'
				},
				{ code: true, name: 'mmproj', note: 'Projectors that let a model see and hear' },
				{ name: 'Hugging Face', note: 'Where open weights are shared, and converted to GGUF' }
			],
			who: 'What it knows'
		}
	];
</script>

{#snippet list(parts: Part[])}
	<ul>
		{#each parts as part (part.name)}
			<li class={['part rounded-panel', part.open ? 'open' : 'bg-surface-lifted elevated-control']}>
				<b class:code={part.code}>{part.name}</b>

				<span>{part.note}</span>
			</li>
		{/each}
	</ul>
{/snippet}

<div class="stack">
	{#each layers as layer, i (layer.name)}
		<section class={['layer card-surface rounded-card', { first: i === 0 }]}>
			<header>
				<span class="n">0{layers.length - i}</span>

				<h3>{layer.name}</h3>

				<span class="who">{layer.who}</span>
			</header>

			{@render list(layer.parts)}

			{#if layer.harness}
				<div class="harness rounded-panel">
					<p class="sub">Harness <span>the agentic loop and the tools it calls</span></p>

					{@render list(layer.harness)}
				</div>
			{/if}
		</section>

		{#if i < layers.length - 1}<span aria-hidden="true" class="arrow">↓</span>{/if}
	{/each}
</div>

<style>
	.stack {
		display: grid;
		justify-items: stretch;
		gap: 6px;
	}

	.layer {
		display: grid;
		gap: 12px;
		padding: 16px;
	}

	header {
		display: flex;
		align-items: baseline;
		gap: 10px;
	}

	.n {
		color: var(--figure-muted);
		font: 500 12px var(--mono);
	}

	.first .n {
		color: var(--art-cinder);
	}

	h3 {
		margin: 0;
		font-size: 17px;
		font-weight: 600;
	}

	.who {
		margin-left: auto;
		color: var(--figure-muted);
		font-size: 13px;
	}

	ul {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.part {
		display: grid;
		gap: 4px;
		align-content: start;
		margin: 0;
		padding: 10px 12px;
	}

	.open {
		border: 1.5px dashed color-mix(in oklab, var(--fg-muted) 55%, transparent);
		opacity: 0.75;
	}

	b {
		font-size: 14px;
		font-weight: 600;
	}

	b.code {
		font: 600 13px var(--mono);
	}

	.part span {
		color: var(--figure-muted);
		font-size: 12px;
		line-height: 1.4;
	}

	.harness {
		display: grid;
		gap: 8px;
		padding: 10px;
		border: 1px solid var(--line-default);
	}

	.sub {
		margin: 0;
		font-size: 14px;
		font-weight: 600;
	}

	.sub span {
		margin-left: 6px;
		color: var(--figure-muted);
		font-weight: 400;
	}

	.arrow {
		justify-self: center;
		color: var(--figure-muted);
		line-height: 1;
	}
</style>
