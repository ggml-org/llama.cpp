<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Downloadable from './communications/posts/Downloadable.svelte';
	import Gpu from './communications/posts/motifs/Gpu.svelte';
	import {
		FORMATS,
		type Ground,
		microsoftStore,
		type MicrosoftStoreCopy
	} from './communications/posts/post';
	import Variants from './communications/posts/Variants.svelte';
	import StoreArt from './communications/store/StoreArt.svelte';
	import StoreScreenshot from './communications/store/StoreScreenshot.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import SyntaxHighlightedCode from '$lib/components/app/content/SyntaxHighlightedCode.svelte';

	// The Microsoft Store listing: the logos (the app tile and the poster art), the super hero art at the top of the
	// listing, and the screenshots: each one practical thing the app does on Windows, a centred headline over Llama for
	// Windows as it is (its quick chat with the real Llama Chat from Store scenes, its tray flyout). The Store may lay its
	// own text over the bottom quarter of the 16:9 pieces, so the words keep to the top. The opener comes in several
	// versions, in a carousel, to compare or to A/B test. Every piece downloads as WebP or PNG at its real size, as set
	// in the controls
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications/Microsoft Store'
	});

	const editable = { controls: { disable: false, expanded: true } };
	// a few lines of a script any app could have, pointed at the server Llama runs
	const code = [
		'from openai import OpenAI',
		'',
		'client = OpenAI(',
		'    base_url="http://localhost:8080/v1",',
		'    api_key="not-needed",',
		')'
	].join('\n');
	const size = (f: keyof typeof FORMATS) => `${FORMATS[f].width} × ${FORMATS[f].height}`;
	// the pieces at 1.6 times a two-up gallery's size: two screenshots to a row on a wide screen, one on a narrower
	// one, scaled down below their width
	const gallery = { itemWidth: '55rem', width: '113.5rem' };
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
		<StateGallery itemWidth="27rem" width={gallery.width}>
			<StateGalleryItem label={FORMATS.msTile.name} note={size('msTile')}>
				<Downloadable name="llama-store-app-tile-{tone}"
					><StoreArt kind="tile" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem
				label={FORMATS.msPoster.name}
				note="The Store's main logo · {size('msPoster')}"
			>
				<Downloadable name="llama-store-poster-art-{tone}"
					><StoreArt kind="poster" {overlay} {tone} /></Downloadable
				>
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
		<StateGallery itemWidth="111.5rem" width={gallery.width}>
			<StateGalleryItem
				label={FORMATS.msHero.name}
				note="No words: the Store sets the name over it · {size('msHero')}"
			>
				<Downloadable name="llama-store-super-hero-art-{tone}"
					><StoreArt kind="hero" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>

<Story
	argTypes={{ tone: grounds }}
	args={{ ...microsoftStore, overlay: false, tone: 'day' }}
	name="Screenshots"
	parameters={editable}
>
	{#snippet template(args)}
		{@const { apps, chat, fits, gpu, models, openers, overlay, tone } = args as MicrosoftStoreCopy &
			ArtArgs}
		<StateGallery {...gallery}>
			<StateGalleryItem label="1, the opener" note="{openers.length} versions of the tagline">
				<Variants items={openers} label="Versions of the opener">
					{#snippet children(o)}
						<Downloadable name="llama-store-screenshot-1-opener-{o.id.toLowerCase()}-{tone}">
							<StoreScreenshot headline={o.headline} kind="brand" lede={o.lede} {overlay} {tone} />
						</Downloadable>
					{/snippet}
				</Variants>
			</StateGalleryItem>

			<StateGalleryItem label="2, chat" note="Llama Chat, in the quick chat">
				<Downloadable name="llama-store-screenshot-2-chat-{tone}"
					><StoreScreenshot {...chat} kind="window" {overlay} scene="chat" {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="3, the tray" note="The tray flyout over the taskbar">
				<Downloadable name="llama-store-screenshot-3-tray-{tone}"
					><StoreScreenshot {...fits} kind="desktop" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="4, models" note="The flyout's Models card, close up">
				<Downloadable name="llama-store-screenshot-4-models-{tone}"
					><StoreScreenshot {...models} kind="card" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="5, hardware" note="The GPU motif">
				<Downloadable name="llama-store-screenshot-5-hardware-{tone}"
					><StoreScreenshot {...gpu} kind="art" {overlay} {tone}>
						<div class="fill gpu"><Gpu /></div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="6, other apps" note="llama-server's OpenAI-compatible API">
				<Downloadable name="llama-store-screenshot-6-apps-{tone}"
					><StoreScreenshot {...apps} kind="art" {overlay} {tone}>
						<div class="code">
							<SyntaxHighlightedCode class="p-3" {code} language="python" maxHeight="none" />
						</div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>

<style>
	.gpu {
		width: calc(var(--u) * 30);
		height: 100%;
	}
	/* the app's own highlighted code block, drawn at about two and a half times its size */
	.code {
		zoom: 2.4;
	}
</style>
