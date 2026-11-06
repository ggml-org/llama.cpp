<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Bulletin from './communications/layouts/Bulletin.svelte';
	import Numeral from './communications/layouts/Numeral.svelte';
	import Product from './communications/layouts/Product.svelte';
	import Roster from './communications/layouts/Roster.svelte';
	import Statement from './communications/layouts/Statement.svelte';
	import BranchMotif from './communications/posts/BranchMotif.svelte';
	import Curve from './communications/posts/Curve.svelte';
	import ModelCard from './communications/posts/ModelCard.svelte';
	import {
		type FormatKey,
		FORMATS,
		type Ground,
		GROUNDS,
		modelDrop
	} from './communications/posts/post';
	import StatCard from './communications/posts/StatCard.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' key visuals: layouts built from the modules on the grid, with stand-in copy, each across the
	// formats. An application is one of these with real copy. Pick the ground and turn on the grid in the controls
	const { Story } = defineMeta({
		argTypes: {
			overlay: { control: 'boolean', name: 'Grid' },
			tone: { control: 'select', name: 'Ground', options: Object.keys(GROUNDS) }
		},
		parameters: {
			controls: { expanded: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Key visuals'
	});

	type Args = { overlay: boolean; tone: Ground };

	const yourAI = 'Your AI.\nOn your computer.';
	const twoLines = 'A headline.\nIn two lines.';
	const footer: [string, string] = ['By the llama.cpp team and Hugging Face', 'Free · Open source'];
	const note = (f: FormatKey) => `${FORMATS[f].width} × ${FORMATS[f].height}`;
</script>

<Story args={{ overlay: false, tone: 'light' }} name="Statement">
	{#snippet template(args)}
		{@const { overlay, tone } = args as Args}
		<StateGallery minItemWidth="18rem">
			{#each ['square', 'portrait', 'story', 'poster', 'landscape', 'billboard'] as const as format (format)}
				<StateGalleryItem label={FORMATS[format].name} note={note(format)}>
					<Statement
						{footer}
						{format}
						header={{ brand: 'Llama', url: 'llama.app' }}
						headline={yourAI}
						lede="Open models, running on your own hardware. Nothing leaves the machine."
						{overlay}
						{tone}
					>
						{#snippet motif()}<BranchMotif />{/snippet}
					</Statement>
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ overlay: false, tone: 'hero' }} name="Product">
	{#snippet template(args)}
		{@const { overlay, tone } = args as Args}
		<StateGallery minItemWidth="18rem">
			{#each ['og', 'landscape', 'square', 'portrait', 'story', 'billboard'] as const as format (format)}
				<StateGalleryItem label={FORMATS[format].name} note={note(format)}>
					<Product
						caption="Llama · Your AI, on your computer"
						{format}
						header={{ brand: 'Llama' }}
						headline={yourAI}
						{overlay}
						{tone}
					/>
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ overlay: false, tone: 'light' }} name="Roster">
	{#snippet template(args)}
		{@const { overlay, tone } = args as Args}
		<StateGallery minItemWidth="18rem">
			{#each ['square', 'portrait', 'story', 'landscape'] as const as format (format)}
				<StateGalleryItem label={FORMATS[format].name} note={note(format)}>
					<Roster
						chips={['first', 'second', 'third']}
						{format}
						header={{ brand: 'icon', kicker: 'Who it is from', url: 'llama.app' }}
						headline="A headline in a few words."
						label="The items"
						lede={format === 'square'
							? undefined
							: 'A lede that says what is new and why it matters.'}
						{overlay}
						{tone}
					>
						{#snippet motif()}<BranchMotif />{/snippet}

						{#snippet cards()}
							{#each modelDrop.models.slice(0, format === 'story' ? 1 : 2) as model (model.name)}<ModelCard
									{model}
									size={0.85}
								/>{/each}
						{/snippet}
					</Roster>
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ overlay: false, tone: 'light' }} name="Bulletin">
	{#snippet template(args)}
		{@const { overlay, tone } = args as Args}
		<StateGallery minItemWidth="18rem">
			{#each ['landscape', 'square', 'portrait'] as const as format (format)}
				<StateGalleryItem label={FORMATS[format].name} note={note(format)}>
					<Bulletin
						{footer}
						{format}
						header={{ brand: 'Llama', pill: 'New today', url: 'llama.app' }}
						headline={twoLines}
						lede="A lede beside it, on what is new and where to find it."
						{overlay}
						{tone}
					>
						{#snippet lead()}
							<div class="curve"><div class="fill"><Curve /></div></div>
						{/snippet}

						{#snippet side()}
							<StatCard accent label="tok/s" value="142" /><StatCard label="GB" value="4.6" />
						{/snippet}
					</Bulletin>
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ overlay: false, tone: 'dark' }} name="Numeral">
	{#snippet template(args)}
		{@const { overlay, tone } = args as Args}
		<StateGallery minItemWidth="18rem">
			{#each ['square', 'portrait', 'story', 'poster', 'landscape', 'billboard'] as const as format (format)}
				<StateGalleryItem label={FORMATS[format].name} note={note(format)}>
					<Numeral
						figureLabel="tokens per second, on a laptop"
						{footer}
						{format}
						header={{ brand: 'Llama', url: 'llama.app' }}
						headline="Fast enough to forget it is local."
						lede="gemma-4 E4B at Q4_K_M on an M4 MacBook Air."
						{overlay}
						{tone}
						value="142"
					/>
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<style>
	.curve {
		display: grid;
		height: 100%;
		min-height: calc(var(--u) * 5);
		padding: calc(var(--u) * 0.6);
		box-sizing: border-box;
		border: 2px solid var(--p-line);
		border-radius: calc(var(--u) * 0.55);
		background: var(--p-card);
	}
</style>
