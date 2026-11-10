<!--
	"Like OpenAI, but on your Mac": the chat for you and the API for your apps, both on the one set of models in Llama.
	Two parts side by side, a wire from each down to the models they share. With `arch`, the two uses are apps (cinder)
	and the models are Llama's (the skies), numbered as Architecture numbers its layers.
-->
<script lang="ts">
	import { Braces, Code, FileText, MessageSquare, Terminal } from '@lucide/svelte';
	import gemma from '$lib/assets/model-logos/gemma.svg?raw';
	import gpt from '$lib/assets/model-logos/gpt.svg?raw';
	import qwen from '$lib/assets/model-logos/qwen.svg?raw';
	import { Badge } from '$lib/components/ui/badge';

	let { arch = false }: { arch?: boolean } = $props();

	const clients = [
		{ example: 'Pi', icon: Terminal, label: 'Coding agents' },
		{ example: 'OpenAI-compatible', icon: Code, label: 'Editors' },
		{ example: 'OpenAI-compatible', icon: MessageSquare, label: 'Chat apps' },
		{ example: 'Any OpenAI SDK', icon: Braces, label: 'Your own code' }
	];
	const models = [
		{ logo: qwen, name: 'Qwen3.8', params: '27B' },
		{ logo: gpt, name: 'gpt-oss', params: '20B' },
		{ logo: gemma, name: 'gemma-4', params: 'E4B' }
	];
</script>

<div class={['lf', { arch }]}>
	<div class="panel">
		<div class="uses">
			<section class="part app">
				<p class="kicker">{#if arch}<b>02</b>{/if}For you</p>
				<h4>Chat <span class="muted">· like ChatGPT</span></h4>

				<div class="chat">
					<span class="file"><FileText size={12} /> lease-agreement.pdf</span>
					<p class="ask">Summarize this and flag anything unusual.</p>
					<p class="answer">
						It's a standard 12-month lease. Two things stand out: it renews automatically unless you
						give 90 days' notice, and…
					</p>
				</div>

				<p class="muted">Click “Open chat” in the menu.</p>
			</section>

			<section class="part app">
				<p class="kicker">{#if arch}<b>02</b>{/if}For your apps</p>
				<h4>API <span class="muted">· like the OpenAI API</span></h4>

				<ul class="clients">
					{#each clients as c (c.label)}
						<li><c.icon size={15} /><span>{c.label}</span><code>{c.example}</code></li>
					{/each}
				</ul>

				<p class="muted">Any app that works with OpenAI, at <code>localhost:9931/v1</code></p>
			</section>
		</div>

		<div class="wires"><span class="wire"></span><span class="wire"></span></div>

		<section class="part lib models">
			<p class="kicker">
				{#if arch}<b>01</b>{:else}<i class="dot"></i>{/if}Your models · download once, use everywhere
			</p>

			<ul>
				{#each models as m (m.name)}
					<li>
						<span class="mark">{@html m.logo}</span>{m.name}<Badge variant="outline">{m.params}</Badge>
					</li>
				{/each}
			</ul>
		</section>
	</div>
</div>

<style>
	.uses {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
		gap: 12px;
	}
	.uses .part {
		align-content: start;
		gap: 10px;
		padding: 16px;
	}
	h4 .muted {
		font-weight: 400;
	}
	.chat {
		display: grid;
		gap: 8px;
		padding: 12px;
		border-radius: var(--radius-control);
		background: var(--surface-page);
		font-size: 13px;
	}
	.file {
		display: inline-flex;
		gap: 5px;
		align-items: center;
		justify-self: end;
		padding: 2px 8px;
		border: 1px solid var(--line-default);
		border-radius: 6px;
		color: var(--figure-muted);
		font-size: 12px;
	}
	.ask {
		justify-self: end;
		max-width: 85%;
		margin: 0;
		padding: 6px 10px;
		border-radius: 12px;
		background: var(--surface-muted);
	}
	.answer {
		margin: 0;
	}
	.clients {
		display: grid;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.clients li {
		display: flex;
		gap: 10px;
		align-items: center;
		padding: 8px 10px;
		border-radius: var(--radius-control);
		background: var(--surface-page);
		font-size: 13px;
	}
	.clients li :global(svg) {
		color: var(--figure-muted);
	}
	.clients span {
		flex: 1;
	}
	.clients code {
		color: var(--figure-muted);
		font-size: 11px;
	}
	/* a wire from the middle of each use, down to the models */
	.wires {
		display: grid;
		grid-template-columns: 1fr 1fr;
		margin-block: -8px;
	}
	@container (width < 34rem) {
		.wires {
			grid-template-columns: 1fr;
		}
		.wires .wire + .wire {
			display: none;
		}
	}
	.models {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 16px;
		align-items: center;
		justify-content: space-between;
	}
	.models ul {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.models li {
		display: inline-flex;
		gap: 7px;
		align-items: center;
		padding: 3px 8px 3px 3px;
		border-radius: var(--radius-control);
		background: var(--surface-page);
		font-size: 13px;
	}
</style>
