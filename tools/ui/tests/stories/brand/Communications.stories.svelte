<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Boards from './communications/Boards.svelte';
	import PaletteRow from './communications/PaletteRow.svelte';
	import { basePalette } from './communications/palettes';
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

	// Communications (social posts, link previews, posters, the site): the post templates, drawn at their real size
	// with their copy in the controls; the neutral key visuals with the real UI on them; and the colour artwork
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Communications'
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

<Story asChild beforeEach={() => seedChatStory()} name="Key visuals">
	<StateGallery columns={1}>
		<StateGalleryItem label={basePalette.name} note={basePalette.note}>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="Artwork">
	<StateGallery columns={1}>
		<StateGalleryItem
			label="Specimen boards and posters"
			note="After the classic Swiss specimen boards, in the brand palette: the art colours, beige and cinder"
		>
			<Boards />
		</StateGalleryItem>
	</StateGallery>
</Story>
