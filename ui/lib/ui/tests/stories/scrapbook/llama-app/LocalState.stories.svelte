<script lang="ts" module>
	import { FAMILIES } from './catalog';
	import LocalCard from './LocalCard.svelte';
	import LocalSizeRow from './LocalSizeRow.svelte';
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// how llama.app says a model is already in Llama on this computer, once the site is connected to it: downloaded, or
	// loaded and ready to chat. Proposals for the catalog card and for a size on the model page; each shows both states
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/UI/llama.app/On this computer'
	});

	const GEMMA = FAMILIES.find((family) => family.name === 'Gemma 4') ?? FAMILIES[0];
	const SIZE = GEMMA.sizes.find((size) => size.builds.length > 1) ?? GEMMA.sizes[0];

	const CARDS = [
		{ note: 'a status Badge where the release month was', title: 'A · Badge', variant: 'badge' },
		{ note: 'a lamp on the avatar’s corner', title: 'B · Avatar lamp', variant: 'avatar' },
		{ note: 'a lamp and a word in the sizes line', title: 'C · Sizes line', variant: 'line' },
		{ note: 'a check before the capability icons', title: 'D · Corner check', variant: 'corner' }
	] as const;

	const ROWS = [
		{ note: 'a lamp and a word under the name, as the Models Manager', title: 'A · Status line', variant: 'status' },
		{ note: 'a status Badge beside the name', title: 'B · Badge', variant: 'badge' },
		{ note: 'the build’s Install button gives way to its state and Chat', title: 'C · On the build', variant: 'build' },
		{ note: 'the status line and the state on the build', title: 'D · Both', variant: 'both' }
	] as const;
</script>

<Story asChild name="Card">
	<div class="min-h-screen bg-background">
		{#each CARDS as card (card.variant)}
			<StateGallery columns={2} title="{card.title} — {card.note}" width="48rem">
				<StateGalleryItem label="Downloaded">
					<LocalCard family={GEMMA} state="downloaded" variant={card.variant} />
				</StateGalleryItem>
				<StateGalleryItem label="Loaded">
					<LocalCard family={GEMMA} state="loaded" variant={card.variant} />
				</StateGalleryItem>
			</StateGallery>
		{/each}
	</div>
</Story>

<Story asChild name="Size row">
	<div class="min-h-screen bg-background">
		{#each ROWS as row (row.variant)}
			<StateGallery columns={1} title="{row.title} — {row.note}" width="56rem">
				<StateGalleryItem label="Downloaded">
					<LocalSizeRow org={GEMMA.org} size={SIZE} state="downloaded" variant={row.variant} />
				</StateGalleryItem>
				<StateGalleryItem label="Loaded">
					<LocalSizeRow org={GEMMA.org} size={SIZE} state="loaded" variant={row.variant} />
				</StateGalleryItem>
			</StateGallery>
		{/each}
	</div>
</Story>
