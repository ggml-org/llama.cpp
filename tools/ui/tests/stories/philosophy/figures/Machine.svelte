<!--
	Local: everything that matters inside one computer. The conversations in Llama Chat, llama-server running
	the model, and the model file you chose, inside a boundary; outside it, only the download the model came from. The
	boundary's label carries the cinder dot.
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';

	const chats = ['A letter to the landlord', 'What is in this photo?', 'Plan for Saturday'];
</script>

<div class="scene">
	<div class="machine">
		<span class="boundary"><i class="dot"></i>Your computer</span>

		<section class="part card-surface rounded-card">
			<header><b>Llama Chat</b><span>Your conversations, saved here</span></header>

			<ul>
				{#each chats as chat (chat)}<li class="rounded-panel bg-surface-lifted">{chat}</li>{/each}
			</ul>
		</section>

		<section class="part card-surface rounded-card">
			<header><b>llama-server</b><span>Runs the model, here</span></header>

			<p class="status"><i class="live"></i><code>localhost:8080</code></p>
		</section>

		<section class="part card-surface rounded-card">
			<header><b>The model</b><span>The file you chose, unchanged</span></header>

			<p class="file"><code>model.Q4_K_M.gguf</code><Badge variant="outline">4.6 GB</Badge></p>
		</section>
	</div>

	<div class="outside">
		<span class="arrow">←</span>

		<p><b>Hugging Face</b><span>Where the model came from, once</span></p>
	</div>
</div>

<style>
	.scene {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 11rem;
		gap: 16px;
		align-items: end;
	}

	.machine {
		position: relative;
		display: grid;
		gap: 8px;
		padding: 28px 16px 16px;
		border: 1.5px dashed var(--line-default);
		border-radius: 18px;
	}

	.boundary {
		position: absolute;
		top: -10px;
		left: 16px;
		display: inline-flex;
		gap: 6px;
		align-items: center;
		padding: 0 8px;
		background: color-mix(in oklab, var(--fg-default) 2.5%, var(--surface-page));
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--art-cinder);
	}

	.part {
		display: grid;
		gap: 10px;
		padding: 14px 16px;
	}

	header {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		align-items: baseline;
		justify-content: space-between;
	}

	header span,
	.outside span {
		color: var(--figure-muted);
		font-size: 13px;
	}

	ul {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	li {
		padding: 4px 10px;
		font-size: 13px;
	}

	p {
		display: flex;
		gap: 8px;
		align-items: center;
		margin: 0;
	}

	code {
		font: 500 13px var(--mono);
	}

	.live {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--status-success, var(--art-green));
	}

	.outside {
		display: grid;
		gap: 8px;
		padding-bottom: 18px;
		color: var(--figure-muted);
	}

	.outside p {
		display: grid;
		gap: 2px;
		padding: 12px 14px;
		border: 1px solid var(--line-default);
		border-radius: 12px;
		color: var(--fg-default);
	}

	.arrow {
		font-size: 18px;
	}

	@media (max-width: 640px) {
		.scene {
			grid-template-columns: minmax(0, 1fr);
		}

		.arrow {
			rotate: 90deg;
			justify-self: start;
		}
	}
</style>
