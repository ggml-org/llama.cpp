<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import DotGrid from './communications/posts/DotGrid.svelte';
	import Downloadable from './communications/posts/Downloadable.svelte';
	import Figure from './communications/posts/Figure.svelte';
	import Laptop from './communications/posts/motifs/Laptop.svelte';
	import { type Ground, macAppStore, type MacAppStoreCopy } from './communications/posts/post';
	import Variants from './communications/posts/Variants.svelte';
	import StoreScreenshot from './communications/store/StoreScreenshot.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import SyntaxHighlightedCode from '$lib/components/app/content/SyntaxHighlightedCode.svelte';

	// The Mac App Store listing: screenshots at 16:10 (2880 × 1800, a 1440 × 900 screen at 2x), each one thing Llama for
	// Mac does, drawn after the app as it is in light mode: its menu (rebuilt from its source) dropped from the menu
	// bar, Llama Chat in a Safari web app's window, the menu's QR code beside a phone that scanned it. The opener and
	// the chat come in several versions, side by side in a row that scrolls. Every piece downloads as WebP or PNG at its
	// real size, as set in the controls
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications/Mac App Store'
	});

	const editable = { controls: { disable: false, expanded: true } };
	// the pieces as the Microsoft Store's: two to a row on a wide screen, one on a narrower one
	const gallery = { itemWidth: '55rem', width: '113.5rem' };
	const grounds = {
		control: 'inline-radio' as const,
		options: ['day', 'night', 'light', 'dark', 'sand']
	};
	// a few lines of a script any app could have, pointed at the server Llama runs
	const code = [
		'from openai import OpenAI',
		'',
		'client = OpenAI(',
		'    base_url="http://localhost:9931/v1",',
		'    api_key="not-needed",',
		')'
	].join('\n');

	interface ArtArgs {
		overlay: boolean;
		tone: Ground;
	}
</script>

<Story
	argTypes={{ tone: grounds }}
	args={{ ...macAppStore, overlay: false, tone: 'day' }}
	name="Screenshots"
	parameters={editable}
>
	{#snippet template(args)}
		{@const { apps, chats, hardware, menu, openers, overlay, phone, size, tone } =
			args as MacAppStoreCopy & ArtArgs}
		<StateGallery {...gallery}>
			<StateGalleryItem full label="1, the opener" note="{openers.length} versions of the tagline">
				<Variants itemWidth={gallery.itemWidth} items={openers} label="Versions of the opener">
					{#snippet children(o)}
						<Downloadable name="llama-mac-screenshot-1-opener-{o.id.toLowerCase()}-{tone}">
							<StoreScreenshot
								headline={o.headline}
								kind="brand"
								lede={o.lede}
								{overlay}
								platform="mac"
								{tone}
							/>
						</Downloadable>
					{/snippet}
				</Variants>
			</StateGalleryItem>

			<StateGalleryItem label="2, the menu bar" note="The menu, rebuilt from the app's source">
				<Downloadable name="llama-mac-screenshot-2-menu-bar-{tone}"
					><StoreScreenshot
						{...menu}
						kind="desktop"
						{overlay}
						platform="mac"
						{tone}
					/></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="3, phone" note="The menu's QR code and the phone that scanned it">
				<Downloadable name="llama-mac-screenshot-3-phone-{tone}"
					><StoreScreenshot {...phone} kind="phone" {overlay} platform="mac" {tone} /></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem
				full
				label="4, chat"
				note="{chats.length} conversations in a Safari web app"
			>
				<Variants itemWidth={gallery.itemWidth} items={chats} label="Versions of the chat">
					{#snippet children(c)}
						<Downloadable name="llama-mac-screenshot-4-chat-{c.id.toLowerCase()}-{tone}">
							<StoreScreenshot
								headline={c.headline}
								kind="window"
								lede={c.lede}
								{overlay}
								platform="mac"
								scene={c.scene}
								{tone}
							/>
						</Downloadable>
					{/snippet}
				</Variants>
			</StateGalleryItem>

			<StateGalleryItem label="5, hardware" note="The laptop motif">
				<Downloadable name="llama-mac-screenshot-5-hardware-{tone}"
					><StoreScreenshot {...hardware} kind="art" {overlay} platform="mac" {tone}>
						<div class="fill laptop"><Laptop /></div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="6, other apps" note="llama-server's OpenAI-compatible API">
				<Downloadable name="llama-mac-screenshot-6-apps-{tone}"
					><StoreScreenshot {...apps} kind="art" {overlay} platform="mac" {tone}>
						<div class="code">
							<SyntaxHighlightedCode class="p-3" {code} language="python" maxHeight="none" />
						</div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>

			<StateGalleryItem label="7, size" note="The app's size, from its README">
				<Downloadable name="llama-mac-screenshot-7-size-{tone}"
					><StoreScreenshot {...size} kind="art" {overlay} platform="mac" {tone}>
						<div class="figure">
							<div class="field"><DotGrid at={null} /></div>

							<Figure accent label={size.figureLabel} size={4.4} value={size.value} />
						</div>
					</StoreScreenshot></Downloadable
				>
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>

<style>
	.laptop {
		width: calc(var(--u) * 26);
		height: 100%;
	}
	/* the app's own highlighted code block, at about two and a half times its size on the larger Mac format */
	.code {
		zoom: 3.6;
	}
	/* the figure over a dot grid, as the Numeral key visual has it */
	.figure {
		position: relative;
		display: grid;
		place-items: center;
		width: 100%;
		height: 100%;
	}
	.figure .field {
		position: absolute;
		inset: 0 calc(var(--u) * 6);
	}
	.figure :global(.figure) {
		position: relative;
		justify-items: center;
	}
</style>
