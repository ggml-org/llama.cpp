<script lang="ts" module>
	import { type Family, FAMILIES } from './catalog';
	import Compare from './Compare.svelte';
	import DiscoverCard from './DiscoverCard.svelte';
	import DiscoverDetails from './DiscoverDetails.svelte';
	import { HUB_MODELS } from './hub';
	import LlamaAppModelCard from './LlamaAppModelCard.svelte';
	import ProposalModelPage from './ProposalModelPage.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Chat's Discover beside llama.app's proposed catalog, on the same models: Discover's side from the Hugging
	// Face API (hub.ts), the catalog's from llama.app's catalog (catalog.ts)
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/UI/llama.app/Discover vs catalog'
	});

	const familyOf = (repo: string): Family =>
		FAMILIES.find((family) => family.sizes.some((size) => size.builds.some((build) => build.repo === repo)))!;

	const PAIRS = HUB_MODELS.map((model) => ({ family: familyOf(model.id), model }));
	const FAMILIES_SHOWN = [...new Set(PAIRS.map((pair) => pair.family))];
	const GEMMA_E4B = PAIRS.find((pair) => pair.model.id === 'ggml-org/gemma-4-E4B-it-GGUF')!;

	const CARD_NOTES = [
		'A Discover card is one repo on the Hub: one size and its quants. A catalog card is a family, its sizes in one line ("E2B–31B · 5 sizes").',
		'Discover names capabilities in labelled badges under the name; the catalog shows them as icons at the top right, each named in its tooltip.',
		'Both work capabilities out from the Hub the same way (pipeline tag, mmproj file, chat template): Discover when it lists a repo, llama.app when it builds. A catalog card shows what any of the family’s sizes can do.',
		'Both lead with the base model’s avatar and ggml-org’s in the corner. Discover’s foot ends with the Hub’s downloads, the catalog’s with the release month; the catalog has no context length yet, Discover has no memory figure.'
	];

	const DETAILS_NOTES = [
		'Discover details one repo (Gemma 4 E4B); the catalog’s page holds the whole family, a quant panel per size.',
		'Discover’s header names the repo and its base model and gives the Hub’s downloads, likes and last update; the catalog’s names the family and gives its release, its sizes and the memory the smallest needs.',
		'Discover’s chips are model size, context, architecture, chat template and license, from the GGUF; the catalog’s are the parameter span and the license.',
		'The quant panel is the same component. In Discover a chip downloads into the app; on llama.app it installs in Llama on a Mac, and a row’s label shows the llama.cpp command in both.'
	];
</script>

<Story asChild name="Cards">
	<Compare notes={CARD_NOTES}>
		{#snippet app()}
			<div class="grid grid-cols-[repeat(auto-fill,minmax(18rem,1fr))] gap-3 p-4">
				{#each PAIRS as pair (pair.model.id)}
					<DiscoverCard family={pair.family} model={pair.model} />
				{/each}
			</div>
		{/snippet}

		{#snippet site()}
			<div class="grid grid-cols-[repeat(auto-fill,minmax(18rem,1fr))] gap-3 p-4">
				{#each FAMILIES_SHOWN as family (family.name)}
					<LlamaAppModelCard {family} href="#{family.name}" />
				{/each}
			</div>
		{/snippet}
	</Compare>
</Story>

<Story asChild name="Details and model page">
	<Compare notes={DETAILS_NOTES}>
		{#snippet app()}
			<DiscoverDetails family={GEMMA_E4B.family} model={GEMMA_E4B.model} />
		{/snippet}

		{#snippet site()}
			<ProposalModelPage family={GEMMA_E4B.family} />
		{/snippet}
	</Compare>
</Story>
