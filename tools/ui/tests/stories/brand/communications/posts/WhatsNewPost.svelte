<!--
	What's new, on the Bulletin key visual: the Llama lockup with a "new today" pill, the headline, a family of models
	in one card with its members inset, the other models with their figures in cinder beside them, and the footer. At
	16:9 the lede is left out: the cards name every model it lists, and at phone sizes there is no room for both. The
	cards keep their names, tags, figures and descriptions, and leave the repo ids to the link; the family's members
	show their names and modalities, the family card itself saying what they are.
-->
<script lang="ts">
	import Bulletin from '../layouts/Bulletin.svelte';
	import ModelCard from './ModelCard.svelte';
	import type { WhatsNewCopy } from './post';

	interface Props {
		copy: WhatsNewCopy;
		overlay?: boolean;
		tone: 'dark' | 'light';
	}

	let { copy, overlay, tone }: Props = $props();
</script>

<Bulletin
	footer={copy.footer}
	format="landscape"
	header={{ brand: 'Llama', pill: copy.pill, url: copy.url }}
	headline={copy.headline}
	{overlay}
	{tone}
>
	{#snippet lead()}
		<ModelCard
			aside={copy.family.aside}
			model={{ about: copy.family.about, name: copy.family.name }}
			size={1.1}
		>
			<div class="family">
				{#each copy.family.models as model (model.name)}<ModelCard
						model={{ badge: model.badge, name: model.name }}
						size={0.85}
						variant="inset"
					/>{/each}
			</div>
		</ModelCard>
	{/snippet}

	{#snippet side()}
		{#each copy.models as model (model.name)}<ModelCard
				model={{ ...model, repo: undefined }}
				size={1}
			/>{/each}
	{/snippet}
</Bulletin>

<style>
	.family {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: calc(var(--u) * 0.36);
		align-self: stretch;
		margin-top: calc(var(--u) * 0.36);
	}
</style>
