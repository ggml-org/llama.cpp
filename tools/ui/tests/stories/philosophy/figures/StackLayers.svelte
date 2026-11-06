<!--
	The Llama stack across its three layers, top to bottom: the application people use, inference that runs the model,
	and the model's open weights, each with what it is for and its real parts. The harness (the agentic loop and the
	tools it calls) is laid across the line between the application and inference, each part on the side it runs on:
	the loop and browser tools in the app, server tools in llama-server, MCP reached from either. Code names are set in
	Llama Mono; the slot for apps that are not ours is dashed; the application, the layer people touch, carries the
	cinder mark.
-->
<script lang="ts">
	interface Part {
		/** a slot for someone else's work, dashed */
		open?: boolean;
		code?: boolean;
		name: string;
		note: string;
	}

	// the harness, laid across the line between the application and inference: each part on the side it runs on
	const loop: Part = {
		name: 'Agentic loop',
		note: 'In Llama Chat: calls tools and feeds back their results'
	};
	const browser: Part = { name: 'Browser tools', note: 'Run in the page, on your device' };
	const server: Part = { name: 'Server tools', note: 'Run by llama-server, next to the model' };
	const mcp: Part = {
		code: true,
		name: 'MCP',
		note: 'Tools and resources from the servers you add, called from either side'
	};

	const layers: { name: string; parts: Part[]; who: string }[] = [
		{
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

{#snippet tile(part: Part, area = '')}
	<li
		style:grid-area={area || null}
		class={['part rounded-panel', part.open ? 'open' : 'bg-surface-lifted elevated-control']}
	>
		<b class:code={part.code}>{part.name}</b>

		<span>{part.note}</span>
	</li>
{/snippet}

{#snippet list(parts: Part[])}
	<ul>
		{#each parts as part (part.name)}{@render tile(part)}{/each}
	</ul>
{/snippet}

<div class="stack">
	{#each layers as layer, i (layer.name)}
		<section
			class={[
				'layer card-surface rounded-card',
				{ above: i === 0, below: i === 1, first: i === 0 }
			]}
		>
			<header>
				<span class="n">0{layers.length - i}</span>

				<h3>{layer.name}</h3>

				<span class="who">{layer.who}</span>
			</header>

			{@render list(layer.parts)}
		</section>

		{#if i === 0}
			<div class="harness card-surface rounded-card">
				<p class="sub">
					Harness <span>The agentic loop and the tools it calls, across the app and the server</span
					>
				</p>

				<ul class="bridge">
					<li class="side app">In the app</li>

					{@render tile(loop, 'loop')}

					{@render tile(browser, 'browser')}

					{@render tile(mcp, 'mcp')}

					<li class="side server">In llama-server</li>

					{@render tile(server, 'server')}
				</ul>
			</div>
		{:else if i < layers.length - 1}
			<span aria-hidden="true" class="arrow">↓</span>
		{/if}
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

	/* the two layers the harness bridges leave room for it to overlap them */
	.above {
		padding-bottom: 88px;
	}

	.below {
		padding-top: 88px;
	}

	.harness {
		position: relative;
		z-index: 1;
		display: grid;
		gap: 10px;
		margin: -76px 28px;
		padding: 12px 14px;
		border: 1px solid var(--line-default);
	}

	.sub {
		margin: 0;
		font-size: 15px;
		font-weight: 600;
	}

	.sub span {
		margin-left: 6px;
		color: var(--figure-muted);
		font-size: 13px;
		font-weight: 400;
	}

	.bridge {
		grid-template-columns: 7rem repeat(3, minmax(0, 1fr));
		grid-template-areas:
			'app-label loop browser mcp'
			'server-label server server mcp';
	}

	.side {
		align-self: center;
		margin: 0;
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}

	.side.app {
		grid-area: app-label;
	}

	.side.server {
		grid-area: server-label;
	}

	@media (width < 720px) {
		.above {
			padding-bottom: 40px;
		}

		.below {
			padding-top: 40px;
		}

		.harness {
			margin: -28px 8px;
		}

		.bridge {
			grid-template-columns: minmax(0, 1fr);
			grid-template-areas: 'app-label' 'loop' 'browser' 'server-label' 'server' 'mcp';
		}
	}

	.arrow {
		justify-self: center;
		color: var(--figure-muted);
		line-height: 1;
	}
</style>
