<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import LinkPreview from './communications/posts/LinkPreview.svelte';
	import ModelDropPost from './communications/posts/ModelDropPost.svelte';
	import {
		FORMATS,
		linkPreview,
		type LinkPreviewCopy,
		modelDrop,
		type ModelDropCopy,
		whatsNew,
		type WhatsNewCopy
	} from './communications/posts/post';
	import WhatsNewPost from './communications/posts/WhatsNewPost.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' applications: key visuals with real copy, the pieces that ship. Model drop is a Roster, What's new
	// a Bulletin, llama.app's link preview a Product and llama.cpp's a Statement. The copy is in the controls
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications'
	});

	const editable = { controls: { disable: false, expanded: true } };
	const tones = ['light', 'dark'] as const;
	const size = (f: keyof typeof FORMATS) => `${FORMATS[f].width} × ${FORMATS[f].height}`;
</script>

<Story args={{ ...modelDrop, overlay: false }} name="Model drop" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as ModelDropCopy & { overlay: boolean }}
		<StateGallery minItemWidth="20rem">
			{#each ['portrait', 'square', 'story'] as const as format (format)}
				{#each tones as tone (tone)}
					<StateGalleryItem label="{FORMATS[format].name}, {tone}" note="Roster · {size(format)}">
						<ModelDropPost {copy} {format} overlay={copy.overlay} {tone} />
					</StateGalleryItem>
				{/each}
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ ...whatsNew, overlay: false }} name="What's new" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as WhatsNewCopy & { overlay: boolean }}
		<StateGallery columns={1}>
			{#each tones as tone (tone)}
				<StateGalleryItem
					label="{FORMATS.landscape.name}, {tone}"
					note="Bulletin · {size('landscape')}"
				>
					<WhatsNewPost {copy} overlay={copy.overlay} {tone} />
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ ...linkPreview, overlay: false }} name="Link previews" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as LinkPreviewCopy & { overlay: boolean }}
		<StateGallery minItemWidth="26rem">
			{#each ['hero', 'navy'] as const as tone (tone)}
				<StateGalleryItem
					label="llama.app, {FORMATS.og.name}, {tone}"
					note="Product · {size('og')}"
				>
					<LinkPreview {copy} kind="og" overlay={copy.overlay} {tone} />
				</StateGalleryItem>
			{/each}

			{#each tones as tone (tone)}
				<StateGalleryItem
					label="llama.cpp, {FORMATS.repo.name}, {tone}"
					note="Statement · {size('repo')}"
				>
					<LinkPreview {copy} kind="repo" overlay={copy.overlay} {tone} />
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>
