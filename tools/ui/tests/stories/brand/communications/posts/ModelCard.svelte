<!--
	A model in a post: an optional tag (cinder for the one that leads) or a line about its maker, its name, a short
	description, cinder stat figures, Mono facts and its repo id. Nested cards (a family) go in children.
-->
<script lang="ts">
	import type { Model } from './post';
	import PostChip from './PostChip.svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		/** a line at the top right, for a family card */
		aside?: string;
		children?: Snippet;
		model: Model;
		/** the name's font size in units; the rest follows it */
		size?: number;
		variant?: 'card' | 'inset';
	}

	let { aside, children, model, size = 1, variant = 'card' }: Props = $props();
</script>

<!-- a repo id breaks after its owner, if it must break at all -->
{#snippet repo(id: string)}{id.slice(0, id.indexOf('/') + 1)}<wbr />{id.slice(
		id.indexOf('/') + 1
	)}{/snippet}

<article style:--s="calc(var(--u) * {size})" class="card {variant}">
	{#if model.tag}
		<PostChip size={Math.max(size * 0.46, 0.54)} variant={model.accent ? 'accent' : 'solid'}
			>{model.tag}</PostChip
		>
	{/if}

	{#if model.about || aside}
		<p class="about">
			<span>{model.about}</span>{#if aside}<span>{aside}</span>{/if}
		</p>
	{/if}

	<h3 class="name">
		<span>{model.name}</span>{#if model.badge}<span class="badge">{model.badge}</span>{/if}
	</h3>

	{#if model.stats?.length}
		<p class="stats">
			{#each model.stats as stat (stat.value)}<span><b>{stat.value}</b> {stat.label}</span>{/each}
		</p>
	{/if}

	{#if model.description}<p class="description">{model.description}</p>{/if}

	{#if model.meta?.length || (model.repo && !model.stats?.length)}
		<p class="meta">
			<!-- each fact kept whole, so a line only breaks between them -->
			{#if model.meta?.length}<span
					>{#each model.meta as fact, i (fact)}{i ? ' · ' : ''}<span class="fact">{fact}</span
						>{/each}</span
				>{/if}

			{#if model.repo}<span>{@render repo(model.repo)}</span>{/if}
		</p>
	{:else if model.repo}
		<p class="repo">{@render repo(model.repo)}</p>
	{/if}

	{#if children}{@render children()}{/if}
</article>

<style>
	.card {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: calc(var(--s) * 0.32);
		min-width: 0;
		padding: calc(var(--s) * 0.9) calc(var(--s) * 0.95);
		border: 2px solid var(--p-line);
		border-radius: calc(var(--s) * 0.6);
		background: var(--p-card);
	}
	.inset {
		padding: calc(var(--s) * 0.7) calc(var(--s) * 0.75);
		border-radius: calc(var(--s) * 0.45);
	}
	.card > :global(*) {
		margin: 0;
	}
	.card > :global(.chip) {
		margin-bottom: calc(var(--s) * 0.12);
	}
	.about {
		display: flex;
		justify-content: space-between;
		align-self: stretch;
		gap: 1em;
		margin-bottom: calc(var(--s) * -0.25);
		color: var(--p-muted);
		font-size: max(calc(var(--s) * 0.44), calc(var(--u) * 0.58));
		line-height: 1.3;
	}
	/* a name never breaks: when the card is too narrow for it and its badge, the badge goes under it */
	.name {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		align-self: stretch;
		gap: 0.3em 0.5em;
		font: 700 var(--s) / 1.1 var(--sans);
		letter-spacing: -0.015em;
	}
	.name > span:first-child {
		white-space: nowrap;
	}
	.badge {
		padding: 0.3em 0.55em;
		border-radius: 0.4em;
		background: var(--p-inset);
		color: var(--p-muted);
		font: 400 max(calc(var(--s) * 0.4), calc(var(--u) * 0.5)) / 1 var(--mono);
		letter-spacing: 0;
	}
	.stats {
		display: flex;
		flex-wrap: wrap;
		/* a unit at the least between figures, so two of them never read as one line */
		gap: 0.3em calc(var(--u) * 1.2);
		color: var(--p-ink);
		font-size: max(calc(var(--s) * 0.48), calc(var(--u) * 0.58));
		line-height: 1.2;
	}
	.stats b {
		margin-right: 0.15em;
		color: var(--art-cinder);
		font-size: 1.45em;
		font-weight: 600;
		letter-spacing: -0.01em;
	}
	.description {
		color: var(--p-ink);
		font-size: max(calc(var(--s) * 0.56), calc(var(--u) * 0.62));
		line-height: 1.35;
		text-wrap: pretty;
		/* in a card taller than its content, the meta or repo below it goes to the card's foot */
		margin-bottom: auto;
	}
	.inset .description {
		color: var(--p-muted);
	}
	.meta {
		display: grid;
		gap: 0.45em;
		margin-top: calc(var(--s) * 0.15);
		color: var(--p-muted);
		font: 400 max(calc(var(--s) * 0.42), calc(var(--u) * 0.58)) / 1.1 var(--mono);
	}
	.fact {
		white-space: nowrap;
	}
	.repo {
		align-self: stretch;
		margin-top: calc(var(--s) * 0.1);
		padding: 0.7em 0.9em;
		border-radius: 0.6em;
		background: var(--p-inset);
		color: var(--p-link);
		font: 400 max(calc(var(--s) * 0.4), calc(var(--u) * 0.58)) / 1 var(--mono);
	}
</style>
