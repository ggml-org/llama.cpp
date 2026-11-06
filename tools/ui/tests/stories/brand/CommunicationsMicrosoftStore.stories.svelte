<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import DotGrid from './communications/posts/DotGrid.svelte';
	import Downloadable from './communications/posts/Downloadable.svelte';
	import Figure from './communications/posts/Figure.svelte';
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
	// versions, side by side in a row that scrolls, to compare or to A/B test. Every piece downloads as WebP or PNG at its
	// real size, as set in the controls
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications/Microsoft Store'
	});

	const editable = { controls: { disable: false, expanded: true } };
	// what the Windows app sends about you, from its README's Privacy section
	const sends = [
		['events', 'none'],
		['analytics', 'none'],
		['install ID', 'none']
	];
	// a few lines of a script any app could have, pointed at the server Llama runs
	const code = [
		'from openai import OpenAI',
		'',
		'client = OpenAI(',
		'    base_url="http://localhost:9931/v1",',
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
		{@const {
			apps,
			chats,
			gpu,
			models,
			openers,
			overlay,
			privacy,
			quick,
			speed,
			tone,
			tray,
			verdicts
		} = args as MicrosoftStoreCopy & ArtArgs}
		<StateGallery {...gallery}>
			<StateGalleryItem full label="1, the opener" note="{openers.length} versions of the tagline">
				<Variants itemWidth={gallery.itemWidth} items={openers} label="Versions of the opener">
					{#snippet children(o)}
						<Downloadable name="llama-store-screenshot-1-opener-{o.id.toLowerCase()}-{tone}">
							<StoreScreenshot headline={o.headline} kind="brand" lede={o.lede} {overlay} {tone} />
						</Downloadable>
					{/snippet}
				</Variants>
			</StateGalleryItem>

			<StateGalleryItem full label="2, chat" note="{chats.length} conversations in the quick chat">
				<Variants itemWidth={gallery.itemWidth} items={chats} label="Versions of the chat">
					{#snippet children(c)}
						<Downloadable name="llama-store-screenshot-2-chat-{c.id.toLowerCase()}-{tone}">
							<StoreScreenshot
								headline={c.headline}
								kind="window"
								lede={c.lede}
								{overlay}
								scene={c.scene}
								{tone}
							/>
						</Downloadable>
					{/snippet}
				</Variants>
			</StateGalleryItem>

			<StateGalleryItem label="3, quick chat" note="Alt + Space over any app">
				<Downloadable name="llama-store-screenshot-3-quick-chat-{tone}"
					><StoreScreenshot {...quick} kind="overlay" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="4, the tray" note="The tray flyout over the taskbar">
				<Downloadable name="llama-store-screenshot-4-tray-{tone}"
					><StoreScreenshot {...tray} kind="desktop" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="5, fit" note="A model's versions, each with its fit verdict">
				<Downloadable name="llama-store-screenshot-5-fit-{tone}"
					><StoreScreenshot
						{...verdicts}
						kind="card"
						{overlay}
						{tone}
						tray="details"
					/></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="6, models" note="The flyout's Models card, close up">
				<Downloadable name="llama-store-screenshot-6-models-{tone}"
					><StoreScreenshot {...models} kind="card" {overlay} {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="7, privacy" note="What the app sends, from its README">
				<Downloadable name="llama-store-screenshot-7-privacy-{tone}"
					><StoreScreenshot {...privacy} kind="art" {overlay} {tone}>
						<div class="receipt">
							<p class="receipt-title">What Llama sends about you</p>

							{#each sends as [what, value] (what)}
								<p class="receipt-row"><span>{what}</span><span>{value}</span></p>
							{/each}

							<p class="receipt-row total">
								<span>your prompts and replies</span><span>none</span>
							</p>

							<p class="receipt-note">
								With downloads and update checks, only the app’s version, the Windows build and the
								processor: <span>llama-win/0.14.1 (10.0.26100; x64)</span>
							</p>
						</div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="8, speed" note="Speculative decoding, set up on its own">
				<Downloadable name="llama-store-screenshot-8-speed-{tone}"
					><StoreScreenshot {...speed} kind="art" {overlay} {tone}>
						<div class="speed">
							<div class="field"><DotGrid at={null} /></div>

							<Figure accent label={speed.figureLabel} size={4.4} value={speed.value} />
						</div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="9, hardware" note="The GPU motif">
				<Downloadable name="llama-store-screenshot-9-hardware-{tone}"
					><StoreScreenshot {...gpu} kind="art" {overlay} {tone}>
						<div class="fill gpu"><Gpu /></div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="10, other apps" note="llama-server's OpenAI-compatible API">
				<Downloadable name="llama-store-screenshot-10-apps-{tone}"
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
	/* what the app sends, as a receipt: Mono rows in the ground's card, the line that matters in cinder */
	.receipt {
		display: grid;
		gap: calc(var(--u) * 0.35);
		width: calc(var(--u) * 20);
		padding: calc(var(--u) * 1) calc(var(--u) * 1.2);
		border: 1px solid var(--p-line);
		border-radius: calc(var(--u) * 0.4);
		background: var(--p-card);
		color: var(--p-ink);
		font: 400 calc(var(--u) * 0.62) / 1.4 var(--mono);
	}
	.receipt p {
		margin: 0;
	}
	.receipt-title {
		margin-bottom: calc(var(--u) * 0.3) !important;
		color: var(--p-muted);
		font-size: calc(var(--u) * 0.48);
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}
	.receipt-row {
		display: flex;
		justify-content: space-between;
		gap: calc(var(--u) * 1);
	}
	.receipt-row.total {
		margin-top: calc(var(--u) * 0.3) !important;
		padding-top: calc(var(--u) * 0.5);
		border-top: 1px dashed var(--p-line);
		color: var(--art-cinder);
		font-weight: 600;
	}
	.receipt-note {
		margin-top: calc(var(--u) * 0.4) !important;
		color: var(--p-muted);
		font: 400 calc(var(--u) * 0.42) / 1.5 var(--sans);
	}
	.receipt-note span {
		font-family: var(--mono);
	}
	/* the figure over a dot grid, as the Numeral key visual has it */
	.speed {
		position: relative;
		display: grid;
		place-items: center;
		width: 100%;
		height: 100%;
	}
	.speed .field {
		position: absolute;
		inset: 0 calc(var(--u) * 6);
	}
	.speed :global(.figure) {
		position: relative;
		justify-items: center;
	}
	/* the app's own highlighted code block, drawn at about two and a half times its size */
	.code {
		zoom: 2.4;
	}
</style>
