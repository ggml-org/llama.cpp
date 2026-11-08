<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Receipt from './communications/layouts/Receipt.svelte';
	import Statement from './communications/layouts/Statement.svelte';
	import AgentsDiagram from './communications/posts/AgentsDiagram.svelte';
	import LinkPreview from './communications/posts/LinkPreview.svelte';
	import ModelDropPost from './communications/posts/ModelDropPost.svelte';
	import {
		type AgentsCopy,
		agentsPost,
		FORMATS,
		linkPreview,
		type LinkPreviewCopy,
		modelDrop,
		type ModelDropCopy,
		openModels,
		type ReceiptCopy,
		whatsNew,
		type WhatsNewCopy
	} from './communications/posts/post';
	import WhatsNewPost from './communications/posts/WhatsNewPost.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' applications: key visuals with real copy, the pieces that ship. Model drop is a Roster, What's new
	// a Bulletin, llama.app's link preview a Product, llama.cpp's a Statement, Agents a Statement with the agents
	// diagram for its motif (after Merve's post) and the open models poster a Receipt. The copy is in the controls
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
			{#each ['day', 'night'] as const as tone (tone)}
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

<Story args={{ ...openModels, overlay: false }} name="Open models poster" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as ReceiptCopy & { overlay: boolean }}
		<StateGallery minItemWidth="20rem">
			{#each ['poster', 'portrait', 'square', 'story'] as const as format (format)}
				{#each tones as tone (tone)}
					<StateGalleryItem label="{FORMATS[format].name}, {tone}" note="Receipt · {size(format)}">
						<Receipt
							footer={copy.footer}
							{format}
							header={{ brand: 'Llama', url: 'llama.app' }}
							headline={copy.headline}
							overlay={copy.overlay}
							rows={copy.rows}
							title={copy.title}
							{tone}
						/>
					</StateGalleryItem>
				{/each}
			{/each}
		</StateGallery>
	{/snippet}
</Story>

<Story args={{ ...agentsPost, overlay: false }} name="Agents" parameters={editable}>
	{#snippet template(args)}
		{@const copy = args as AgentsCopy & { overlay: boolean }}
		<StateGallery minItemWidth="28rem">
			{#each tones as tone (tone)}
				<StateGalleryItem label="Landscape 16:9, {tone}" note="Statement · {size('landscape')}">
					<Statement
						command={copy.command}
						commandSize={0.68}
						footer={copy.footer}
						format="landscape"
						header={{ brand: 'Llama', url: copy.url }}
						headline={copy.headline}
						lede={copy.lede}
						ledeSize={0.9}
						overlay={copy.overlay}
						prompt={false}
						split={7}
						{tone}
					>
						{#snippet motif()}<AgentsDiagram
								agents={copy.agents}
								endpoint={copy.endpoint}
							/>{/snippet}
					</Statement>
				</StateGalleryItem>
			{/each}
		</StateGallery>
	{/snippet}
</Story>
