<!--
	A model drop as a carousel, for Instagram and LinkedIn: at phone sizes one image can't hold every model in full, so
	the drop is a cover and then one slide per model. `slide` picks which. 0 is the cover, on the Roster key visual:
	the icon and who it is from, the headline, the lede beside the decision tree, use cases as chips, and what is
	inside, a row per model with its name and tag (the leading one in cinder) and the catalog's URL beside the label; the
	square leaves out the lede and chips, which the model slides make up for. From 1, a model's own slide: the same
	header with its place in the carousel where the URL would be, and the model's card large, every detail at a size a
	phone shows.
-->
<script lang="ts">
	import Roster from '../layouts/Roster.svelte';
	import BranchMotif from './BranchMotif.svelte';
	import ModelCard from './ModelCard.svelte';
	import { FORMATS, type ModelDropCopy } from './post';
	import PostChip from './PostChip.svelte';
	import PostFrame from './PostFrame.svelte';
	import PostHeader from './PostHeader.svelte';

	interface Props {
		copy: ModelDropCopy;
		format: 'portrait' | 'square' | 'story';
		overlay?: boolean;
		/** 0 for the cover, then one per model */
		slide?: number;
		tone: 'dark' | 'light';
	}

	let { copy, format, overlay, slide = 0, tone }: Props = $props();

	const model = $derived(copy.models[slide - 1]);
	// the card's name size in units: as large as the format's height lets every detail stay big
	const size = $derived({ portrait: 2, square: 1.7, story: 2.2 }[format]);
</script>

{#if slide === 0}
	<Roster
		chips={format === 'square' ? [] : copy.chips}
		{format}
		header={{ brand: 'icon', kicker: copy.kicker }}
		headline={copy.headline}
		label={copy.section}
		lede={format === 'square' ? undefined : copy.lede}
		link={copy.url}
		{overlay}
		{tone}
	>
		{#snippet motif()}<BranchMotif />{/snippet}

		{#snippet cards()}
			<ol class="inside">
				{#each copy.models as m (m.name)}
					<li>
						<span class="name">{m.name}</span>

						{#if m.tag}<PostChip variant={m.accent ? 'accent' : 'solid'}>{m.tag}</PostChip>{/if}
					</li>
				{/each}
			</ol>
		{/snippet}
	</Roster>
{:else if model}
	<PostFrame
		format={FORMATS[format]}
		label="Model drop, {model.name}, {FORMATS[format].name}, {tone}"
		{overlay}
		{tone}
	>
		<!-- its place in the carousel sits where a URL would -->
		<PostHeader brand="icon" kicker={copy.kicker} url="{slide + 1} / {copy.models.length + 1}" />

		<div class="slide">
			<div class="label">
				<p class="t-label">{copy.section}</p>

				<span class="link">{copy.url}</span>
			</div>

			<ModelCard {model} {size} />
		</div>
	</PostFrame>
{/if}

<style>
	/* what is inside: a row per model across the width, its name and its tag at either end, between hairlines */
	.inside {
		grid-column: 1 / -1;
		margin: 0;
		padding: 0;
		border-top: 2px solid var(--p-line);
		list-style: none;
	}
	.inside li {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: calc(var(--u) * 2);
		padding: calc(var(--u) * 0.5) 0;
		border-bottom: 2px solid var(--p-line);
	}
	.name {
		font: 700 calc(var(--u) * 1.1) / 1.1 var(--sans);
		letter-spacing: -0.02em;
	}
	.slide {
		display: flex;
		flex: 1;
		flex-direction: column;
		justify-content: flex-end;
		min-height: 0;
		margin-top: calc(var(--u) * 1.2);
	}
	.label {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: calc(var(--u) * 2);
		margin-bottom: calc(var(--u) * 0.45);
	}
	.link {
		color: var(--p-ink);
		font: 400 calc(var(--u) * 0.6) / 1 var(--mono);
	}
</style>
