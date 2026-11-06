<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import HeadlineBlock from './communications/posts/HeadlineBlock.svelte';
	import ModelCard from './communications/posts/ModelCard.svelte';
	import { linkPreview, modelDrop, whatsNew } from './communications/posts/post';
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

	// specimen tiles at their real size, on one fixed unit, sized to what they hold; the galleries fit as many to a row
	// as the width allows
	const U = 28;
	const tile = (width: number, height: number) => ({ height, name: 'Tile', width });
	const rem = (px: number) => `${px / 16}rem`;
	const headline = tile(540, 480);
	const cards = tile(540, 760);
	const family = tile(540, 760);
	const stats = tile(540, 360);
	const shot = tile(500, 640);
</script>

<Story asChild name="Headline blocks">
	<StateGallery minItemWidth={rem(headline.width)}>
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem label="Headline, {tone}" note="Label, headline, lede and chips">
				<PostFrame actual format={headline} label="Headline block" {tone} unit={U}>
					<HeadlineBlock
						chips={modelDrop.chips.slice(0, 2)}
						headline="Decision models, running locally."
						label="Fresh drops"
						lede="Get back a calibrated probability for every option."
					/>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem label="Display, {tone}" note="The headline alone, at display size">
				<PostFrame actual format={headline} label="Headline block" {tone} unit={U}>
					<HeadlineBlock headline={linkPreview.headline} level="display" />
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Model cards">
	<StateGallery minItemWidth={rem(family.width)}>
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Models, {tone}"
				note="Tagged, with stats, and a family with its members inset"
			>
				<PostFrame actual format={cards} label="Model cards" {tone} unit={U}>
					<div class="cards">
						<ModelCard model={modelDrop.models[0]} size={0.9} />

						<ModelCard model={whatsNew.models[1]} size={0.9} />
					</div>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem label="Family, {tone}" note="One card holding its members">
				<PostFrame actual format={family} label="Family card" {tone} unit={U}>
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
	<StateGallery minItemWidth={rem(stats.width)}>
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Stats, {tone}"
				note="Figures that compare, the one that matters in cinder"
			>
				<PostFrame actual format={stats} label="Stat cards" {tone} unit={U}>
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
	<StateGallery minItemWidth={rem(shot.width)}>
		{#each ['day', 'night'] as const as tone (tone)}
			<StateGalleryItem
				label="Product, {tone}"
				note="The app at a width in units, whatever the format"
			>
				<PostFrame actual format={shot} label="Product shot" {tone} unit={U}>
					<div class="shot"><ProductShot size={14} unit={U} /></div>
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
