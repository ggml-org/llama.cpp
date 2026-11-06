<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import HeadlineBlock from './communications/posts/HeadlineBlock.svelte';
	import ModelCard from './communications/posts/ModelCard.svelte';
	import { grid, linkPreview, modelDrop, whatsNew } from './communications/posts/post';
	import PostFrame from './communications/posts/PostFrame.svelte';
	import ProductShot from './communications/posts/ProductShot.svelte';
	import StatCard from './communications/posts/StatCard.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' modules: elements put together into the parts a key visual is made of. Headline blocks, model and
	// stat cards, and the product shot
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Modules'
	});

	const tile = { height: 720, name: 'Tile', width: 1080 };
	const tall = { height: 1080, name: 'Tile', width: 1080 };
</script>

<Story asChild name="Headline blocks">
	<StateGallery minItemWidth="20rem">
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem label="Headline, {tone}" note="Label, headline, lede and chips">
				<PostFrame format={tile} label="Headline block" {tone}>
					<HeadlineBlock
						chips={modelDrop.chips.slice(0, 2)}
						headline="Decision models, running locally."
						label="Fresh drops"
						lede="Get back a calibrated probability for every option."
					/>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem label="Display, {tone}" note="The headline alone, at display size">
				<PostFrame format={tile} label="Headline block" {tone}>
					<HeadlineBlock headline={linkPreview.headline} level="display" />
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Model cards">
	<StateGallery minItemWidth="20rem">
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Models, {tone}"
				note="Tagged, with stats, and a family with its members inset"
			>
				<PostFrame format={tall} label="Model cards" {tone}>
					<div class="cards">
						<ModelCard model={modelDrop.models[0]} size={0.9} />

						<ModelCard model={whatsNew.models[1]} size={0.9} />
					</div>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem label="Family, {tone}" note="One card holding its members">
				<PostFrame format={tall} label="Family card" {tone}>
					<ModelCard
						aside={whatsNew.family.aside}
						model={{ about: whatsNew.family.about, name: whatsNew.family.name }}
						size={1}
					>
						<div class="family">
							{#each whatsNew.family.models as model (model.name)}<ModelCard
									{model}
									size={0.7}
									variant="inset"
								/>{/each}
						</div>
					</ModelCard>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Stat cards">
	<StateGallery minItemWidth="20rem">
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Stats, {tone}"
				note="Figures that compare, the one that matters in cinder"
			>
				<PostFrame format={tile} label="Stat cards" {tone}>
					<div class="stats">
						<StatCard accent label="tok/s" note="On an M4 Max, at Q4_K_M" value="142" />

						<StatCard label="GB" note="Fits in 8 GB of memory" value="4.6" />

						<StatCard label="context" note="Image input included" value="1M" />
					</div>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Product shot">
	<StateGallery minItemWidth="20rem">
		{#each ['hero', 'navy'] as const as tone (tone)}
			<StateGalleryItem
				label="Product, {tone}"
				note="The app at a width in units, whatever the format"
			>
				<PostFrame format={tall} label="Product shot" {tone}>
					<div class="shot"><ProductShot size={14} unit={grid(tall).unit} /></div>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<style>
	.cards {
		display: grid;
		gap: var(--gutter);
	}
	.family {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: calc(var(--u) * 0.36);
		align-self: stretch;
		margin-top: calc(var(--u) * 0.36);
	}
	.stats {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: var(--gutter);
		margin-block: auto;
	}
	.shot {
		display: grid;
		flex: 1;
		place-items: center;
	}
</style>
