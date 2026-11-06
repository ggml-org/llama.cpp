<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Numeral from './communications/layouts/Numeral.svelte';
	import Product from './communications/layouts/Product.svelte';
	import Receipt from './communications/layouts/Receipt.svelte';
	import Statement from './communications/layouts/Statement.svelte';
	import Laptop from './communications/posts/motifs/Laptop.svelte';
	import {
		FORMATS,
		type Ground,
		microsoftStore,
		type MicrosoftStoreCopy
	} from './communications/posts/post';
	import StoreArt from './communications/store/StoreArt.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The Microsoft Store listing: the logos (the app tile and the poster art), the super hero art at the top of the
	// listing, and the screenshots, each a key visual with the app as it shows on Windows. The Store may lay its own
	// text over the bottom quarter of the 16:9 pieces, so they keep it clear, like a story's safe area
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications/Microsoft Store'
	});

	const editable = { controls: { disable: false, expanded: true } };
	const header = { brand: 'Llama', url: 'llama.app' } as const;
	const size = (f: keyof typeof FORMATS) => `${FORMATS[f].width} × ${FORMATS[f].height}`;
	const grounds = {
		control: 'inline-radio' as const,
		options: ['day', 'night', 'light', 'dark', 'sand']
	};

	interface ArtArgs {
		overlay: boolean;
		tone: Ground;
	}
</script>

<Story
	argTypes={{ tone: grounds }}
	args={{ overlay: false, tone: 'day' }}
	name="Store logos"
	parameters={editable}
>
	{#snippet template(args)}
		{@const { overlay, tone } = args as ArtArgs}
		<StateGallery minItemWidth="16rem">
			<StateGalleryItem label={FORMATS.msTile.name} note={size('msTile')}>
				<StoreArt kind="tile" {overlay} {tone} />
			</StateGalleryItem>

			<StateGalleryItem
				label={FORMATS.msPoster.name}
				note="The Store's main logo · {size('msPoster')}"
			>
				<StoreArt kind="poster" {overlay} {tone} />
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>

<Story
	argTypes={{ tone: grounds }}
	args={{ overlay: false, tone: 'night' }}
	name="Super hero art"
	parameters={editable}
>
	{#snippet template(args)}
		{@const { overlay, tone } = args as ArtArgs}
		<StateGallery columns={1}>
			<StateGalleryItem
				label={FORMATS.msHero.name}
				note="No words: the Store sets the name over it · {size('msHero')}"
			>
				<StoreArt kind="hero" {overlay} {tone} />
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ ...microsoftStore, overlay: false }} name="Screenshots" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as MicrosoftStoreCopy & { overlay: boolean }}
		{@const [product, statement, receipt, numeral] = copy.screenshots}
		{@const note = (layout: string) => `${layout} · ${size('msScreenshot')}`}
		<StateGallery minItemWidth="26rem">
			<StateGalleryItem label="1, day" note={note('Product')}>
				<Product
					caption={copy.caption}
					format="msScreenshot"
					{header}
					headline={product.headline}
					lede={product.lede}
					overlay={copy.overlay}
					platform="windows"
					tone="day"
				/>
			</StateGalleryItem>

			<StateGalleryItem label="2, light" note={note('Statement')}>
				<Statement
					format="msScreenshot"
					{header}
					headline={statement.headline}
					lede={statement.lede}
					overlay={copy.overlay}
					tone="light"
				>
					{#snippet motif()}<Laptop />{/snippet}
				</Statement>
			</StateGalleryItem>

			<StateGalleryItem label="3, sand" note={note('Receipt')}>
				<Receipt
					format="msScreenshot"
					{header}
					headline={receipt.headline}
					overlay={copy.overlay}
					rows={receipt.rows}
					title={receipt.title}
					tone="sand"
				/>
			</StateGalleryItem>

			<StateGalleryItem label="4, night" note={note('Numeral')}>
				<Numeral
					figureLabel={numeral.figureLabel}
					format="msScreenshot"
					{header}
					headline={numeral.headline}
					lede={numeral.lede}
					overlay={copy.overlay}
					tone="night"
					value={numeral.value}
				/>
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>
