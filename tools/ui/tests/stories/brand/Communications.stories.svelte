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

	// Communications' post templates (social posts and link previews), drawn at their real size with their copy in the
	// controls. The key visuals and the artwork are their own items in Communications
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Posts'
	});

	const editable = { controls: { disable: false, expanded: true } };
	const tones = ['light', 'dark'] as const;
	const size = (f: keyof typeof FORMATS) => `${FORMATS[f].width} × ${FORMATS[f].height}`;
</script>

<Story args={{ ...modelDrop }} name="Model drop" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as ModelDropCopy}
		<StateGallery minItemWidth="20rem">
			{#each ['portrait', 'square', 'story'] as const as format (format)}
				{#each tones as tone (tone)}
					<StateGalleryItem label="{FORMATS[format].name}, {tone}" note={size(format)}>
						<ModelDropPost {copy} {format} {tone} />
					</StateGalleryItem>
				{/each}
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ ...whatsNew }} name="What's new" parameters={editable}>
	{#snippet template(args)}
		<StateGallery columns={1}>
			{#each tones as tone (tone)}
				<StateGalleryItem label="{FORMATS.landscape.name}, {tone}" note={size('landscape')}>
					<WhatsNewPost copy={args as WhatsNewCopy} {tone} />
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ ...linkPreview }} name="Link previews" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as LinkPreviewCopy}
		<StateGallery minItemWidth="26rem">
			{#each ['hero', 'navy'] as const as tone (tone)}
				<StateGalleryItem label="llama.app, {FORMATS.og.name}, {tone}" note={size('og')}>
					<LinkPreview {copy} kind="og" {tone} />
				</StateGalleryItem>
			{/each}

			{#each tones as tone (tone)}
				<StateGalleryItem label="llama.cpp, {FORMATS.repo.name}, {tone}" note={size('repo')}>
					<LinkPreview {copy} kind="repo" {tone} />
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>
