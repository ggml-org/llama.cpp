<!--
	Proposal A · Same screen: Llama Chat's first screen twice. For most people, a greeting, everyday suggestions and the
	box to type in, with a photo button. For a power user, the same screen with the model named and its settings
	open beside it. Built from the real Button, Badge and token surfaces.
-->
<script lang="ts">
	import { ArrowUp, Camera, SlidersHorizontal } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';

	const asks = [
		'Write a thank-you note to my neighbour',
		'What does this letter from the bank mean?',
		'Plan dinners for the week',
		'What plant is this?'
	];

	const settings = [
		['Model', 'Qwen3.8 27B · Q4_K_M'],
		['Temperature', '0.8'],
		['Top-p', '0.95'],
		['Context', '32,768'],
		['System prompt', 'You are a careful…']
	];
</script>

{#snippet composer(power: boolean)}
	<div class="composer rounded-panel bg-surface-lifted elevated-control">
		<span class="placeholder">Ask anything</span>

		<div class="tools">
			<Button aria-label="Add a photo" size="icon" tabindex={-1} variant="ghost">
				<Camera aria-hidden="true" />
			</Button>

			{#if power}<Badge variant="outline">Qwen3.8 27B</Badge>{/if}

			<Button aria-label="Send" class="send" size="icon" tabindex={-1}>
				<ArrowUp aria-hidden="true" />
			</Button>
		</div>
	</div>
{/snippet}

<div class="same">
	<section class="screen card-surface rounded-card" inert>
		<p class="who first">Most people</p>

		<p class="hello">What can I help with?</p>

		<ul class="asks">
			{#each asks as ask (ask)}<li class="ask rounded-control">{ask}</li>{/each}
		</ul>

		{@render composer(false)}
	</section>

	<section class="screen card-surface rounded-card" inert>
		<p class="who">Power user</p>

		<div class="panel rounded-panel bg-surface-lifted elevated-control">
			<p class="panel-head"><SlidersHorizontal aria-hidden="true" size={14} /> Settings</p>

			<dl>
				{#each settings as [k, v] (k)}
					<dt>{k}</dt>

					<dd>{v}</dd>
				{/each}
			</dl>
		</div>

		{@render composer(true)}
	</section>
</div>

<style>
	.same {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
	}

	.screen {
		display: grid;
		grid-template-rows: auto 1fr auto;
		gap: 12px;
		padding: 16px;
	}

	.screen:first-child {
		grid-template-rows: auto auto 1fr auto;
	}

	p {
		margin: 0;
	}

	.who {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.who.first {
		color: var(--art-cinder);
	}

	.hello {
		font-size: 20px;
		font-weight: 600;
	}

	.asks {
		display: grid;
		gap: 6px;
		align-content: start;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.ask {
		margin: 0;
		padding: 8px 10px;
		border: 1px solid var(--line-default);
		font-size: 13px;
	}

	.composer {
		display: grid;
		gap: 6px;
		padding: 10px 10px 6px 14px;
	}

	.placeholder {
		color: var(--figure-muted);
		font-size: 14px;
	}

	.tools {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.tools :global(.send) {
		margin-left: auto;
	}

	.panel {
		display: grid;
		gap: 8px;
		align-self: start;
		padding: 12px;
	}

	.panel-head {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 13px;
		font-weight: 600;
	}

	dl {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 6px 12px;
		margin: 0;
		font-size: 13px;
	}

	dt {
		color: var(--figure-muted);
	}

	dd {
		margin: 0;
		overflow: hidden;
		font: 500 12px var(--mono);
		text-align: right;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	@media (width < 640px) {
		.same {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
