<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Gpu from './communications/posts/motifs/Gpu.svelte';
	import {
		FORMATS,
		type Ground,
		microsoftStore,
		type MicrosoftStoreCopy
	} from './communications/posts/post';
	import StoreArt from './communications/store/StoreArt.svelte';
	import StoreScreenshot from './communications/store/StoreScreenshot.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The Microsoft Store listing: the logos (the app tile and the poster art), the super hero art at the top of the
	// listing, and the screenshots: each one practical thing the app does on Windows, a centred headline over the real
	// app (Store scenes) running off the bottom edge. The Store may lay its own text over the bottom quarter of the 16:9
	// pieces, so the words keep to the top
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

<Story
	argTypes={{ tone: grounds }}
	args={{ ...microsoftStore, overlay: false, tone: 'day' }}
	name="Screenshots"
	parameters={editable}
>
	{#snippet template(args)}
		{@const { apps, chat, fits, gpu, models, opener, overlay, tone } = args as MicrosoftStoreCopy &
			ArtArgs}
		<StateGallery minItemWidth="26rem">
			<StateGalleryItem label="1, the opener" note={size('msScreenshot')}>
				<StoreScreenshot {...opener} kind="brand" {overlay} {tone} />
			</StateGalleryItem>

			<StateGalleryItem label="2, chat" note="Llama Chat, in its window">
				<StoreScreenshot {...chat} kind="window" {overlay} scene="chat" {tone} />
			</StateGalleryItem>

			<StateGalleryItem label="3, the tray" note="The flyout over the taskbar">
				<StoreScreenshot {...fits} kind="desktop" {overlay} {tone} />
			</StateGalleryItem>

			<StateGalleryItem label="4, models" note="The model manager">
				<StoreScreenshot
					{...models}
					kind="window"
					{overlay}
					scene="models"
					title="Llama · Models"
					{tone}
				/>
			</StateGalleryItem>

			<StateGalleryItem label="5, hardware" note="The GPU motif">
				<StoreScreenshot {...gpu} kind="art" {overlay} {tone}>
					<div class="fill gpu"><Gpu /></div>
				</StoreScreenshot>
			</StateGalleryItem>

			<StateGalleryItem label="6, other apps" note="llama-server's OpenAI-compatible API">
				<StoreScreenshot {...apps} kind="art" {overlay} {tone}>
					<pre class="code"><span class="file">app.py</span>{code}</pre>
				</StoreScreenshot>
			</StateGalleryItem>
		</StateGallery>
	{/snippet}
</Story>

<style>
	.gpu {
		width: calc(var(--u) * 30);
		height: 100%;
	}
	/* a code card: the ground's card in Llama Mono, the address in cinder */
	.code {
		display: grid;
		gap: calc(var(--u) * 0.5);
		margin: 0;
		padding: calc(var(--u) * 1.2) calc(var(--u) * 1.6);
		border: 1px solid var(--p-line);
		border-radius: calc(var(--u) * 0.4);
		background: var(--p-card);
		color: var(--p-ink);
		font: 400 calc(var(--u) * 0.85) / 1.6 var(--mono);
		white-space: pre;
	}
	.file {
		color: var(--p-muted);
		font-size: calc(var(--u) * 0.48);
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}
</style>
