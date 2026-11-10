<script lang="ts" module>
	import { FAMILIES } from './catalog';
	import Frames from './Frames.svelte';
	import LlamaAppModelCard from './LlamaAppModelCard.svelte';
	import ProposalCatalog from './ProposalCatalog.svelte';
	import ProposalModelPage from './ProposalModelPage.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// how llama.app shows models, proposed for its first release on Llama UI: Llama Chat's Discover, its model card
	// and its details, on the website. Built from Model UI and llama.app's catalog. Scrapbook › UI › llama.app › Discover vs catalog puts it beside the app
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/UI/llama.app/Proposal'
	});

	const pick = (...names: string[]) => names.map((name) => FAMILIES.find((family) => family.name === name)!);
	const GEMMA = pick('Gemma 4')[0];
</script>

<!-- the card on its own, in the cases the catalog has: several sizes, reasoning and tools, many sizes, a
     non-commercial decision model, an embedding model -->
<Story asChild name="Model card">
	<Frames>
		<div class="grid grid-cols-[repeat(auto-fill,minmax(18rem,1fr))] gap-3 p-5 @3xl:p-8">
			{#each pick('Gemma 4', 'GPT-OSS', 'Ministral 3', 'Bespoke Nimble', 'EmbeddingGemma 2') as family (family.name)}
				<LlamaAppModelCard {family} href="#{family.name}" />
			{/each}
		</div>
	</Frames>
</Story>

<!-- the models page: each section a grid of cards, Discover's -->
<Story asChild name="Catalog">
	<Frames><ProposalCatalog families={FAMILIES} /></Frames>
</Story>

<!-- a family's page: Discover's details, with every quant an install link -->
<Story asChild name="Model page">
	<Frames><ProposalModelPage family={GEMMA} /></Frames>
</Story>
